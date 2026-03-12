import torch
from experimental_setups.sEM_Qres.mfa import MFA

class MFA_OTFP:
    def __init__(self, init_data: torch.Tensor, n_channels: int, outlier_significance: float, device: str, outlier_update_treshold: int, L2_normalization: bool = True, q_max: int = 5):
        # System hyperparameters
        self.device = device
        self.n_channels = n_channels
        self.q_max = q_max
        self.outlier_update_treshold = outlier_update_treshold
        self.outlier_significance = outlier_significance
        self.L2_normalization = L2_normalization

        # MFA model-state 
        K, q = self._perform_model_selection(data=init_data, n_channels=n_channels, q_max=q_max)
        self.MFA = MFA(n_components=K, n_channels=n_channels, n_factors=q).to(self.device)
        
        self.component_repos = {}
        
        self._run_mfa_setup(init_data)
        
        # Streaming statistics
        self.n_samples_seen = 0
        self.n_model_updates = 0

        # Memory buffers
        self.global_outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32) 
        self.num_outliers_on_shelf = 0
        
        # Local shelves for drifting known materials
        self.local_shelf_counts = torch.zeros(self.MFA.K, dtype=torch.long, device=self.device) 
        self.component_pixel_counts = torch.zeros(self.MFA.K, dtype=torch.long, device=self.device) 
        self.local_shelves = {k: [] for k in range(self.MFA.K)} 

        print(f"Finished setup of OTFP-MFA: K={self.MFA.K}, q={self.MFA.q}")

    def _run_mfa_setup(self, X):
        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        with torch.no_grad():
            self.MFA.initialize_parameters(X)
            self.MFA.fit(X) 
            self.MFA.init_sufficient_statistics(X)
            log_resp_norm, log_likelihood, log_probs = self.MFA.e_step(X)
        
        self.repo_size = 360
        self.component_repos = {}
        self.q_limits = torch.zeros(self.MFA.K, device=self.device)
        
        best_components = torch.argmax(log_resp_norm, dim=1)
        
        # Calculate Q-residuals for all setup data
        Q_res_setup = self.MFA.compute_q_residuals(X)
        
        for k in range(self.MFA.K):
            mask_k = (best_components == k)
            X_k = X[mask_k]
            
            if len(X_k) > 0:
                probs_k = log_probs[mask_k, k]
                num_to_take = min(self.repo_size, len(X_k))
                
                # Anchor pixels are still selected based on pure geometric fit
                top_values, top_indices = torch.topk(probs_k, num_to_take)
                self.component_repos[k] = X_k[top_indices].clone()

                # BUT calculate the Q-limits using ALL pixels that were assigned to this component (X_k)
                Q_cluster_full = Q_res_setup[mask_k, k] 

                med_q = torch.median(Q_cluster_full)
                mad_q = torch.median(torch.abs(Q_cluster_full - med_q))
                mad_q = torch.clamp(mad_q, min=1e-4)
                local_std_q = 1.4826 * mad_q + 1e-6
                
                # Using outlier_significance as the configurable multiplier (e.g., 3.0 or 4.0)
                self.q_limits[k] = (med_q + (self.outlier_significance * local_std_q)).item()
                
            else:
                self.component_repos[k] = torch.empty((0, self.n_channels), device=self.device)
                self.q_limits[k] = float('inf')

        print(f"Q-limits initialized for {self.MFA.K} components.")
        return

    def _perform_model_selection(self, data, n_channels, q_max):
        # Dummy implementation
        K = 2      
        q = 4
        return K, q

    def process_data_block(self, X):
        if self.MFA is None:
            raise RuntimeError("CRITICAL: MFA model was not initialized.")

        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        self.n_samples_seen += X.shape[0]
        
        # --- Calculate Q-residuals for incoming batch ---
        with torch.no_grad():
            Q_res = self.MFA.compute_q_residuals(X)  # Shape: (N, K)

        # 1. A pixel is a global outlier ONLY if it exceeds the Q-limit for ALL components
        outlier_mask = torch.all(Q_res > self.q_limits.unsqueeze(0), dim=1)
        inlier_mask = ~outlier_mask
        
        num_new_outliers = outlier_mask.sum().item()

        # =====================================================================
        # 1. TRACK INLIERS & CATCH LOCAL DRIFTERS (SHADOWS/MIXTURES)
        # =====================================================================
        if inlier_mask.any():
            with torch.no_grad():
                X_inliers = X[inlier_mask]
                Q_res_inliers = Q_res[inlier_mask]

                # Get the base geometric responsibilities
                log_resp_norm, _, _ = self.MFA.e_step(X_inliers)

                # --- Strict Q-Boundary Enforcement ---
                # Zero out responsibilities (set log to -inf) for components where the pixel fails the Q-limit
                valid_assignment_mask = Q_res_inliers <= self.q_limits.unsqueeze(0)
                log_resp_norm[~valid_assignment_mask] = -float('inf')

                # Re-normalize responsibilities so they sum to 1 across the valid sub-spaces
                log_likelihood_inliers = torch.logsumexp(log_resp_norm, dim=1)
                log_resp_norm_adjusted = log_resp_norm - log_likelihood_inliers.unsqueeze(1)

                # Filter out edge-case pixels that somehow got zeroed out entirely 
                valid_pixels = ~torch.isinf(log_likelihood_inliers)

                if valid_pixels.any():
                    self.MFA.stepwise_em_update(X_inliers[valid_pixels], log_resp_norm_adjusted[valid_pixels])
        # =====================================================================

        # =====================================================================
        # 2. GLOBAL OUTLIER SHELF & COMPONENT BIRTHING
        # =====================================================================
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            # Use DBSCAN to isolate true signals from random noise ---
            from sklearn.cluster import DBSCAN
            
            # eps=0.05 on cosine distance means pixels must be highly directionally similar.
            # min_samples ensures we don't build models for tiny micro-clusters.
            dbscan = DBSCAN(eps=0.05, min_samples=2*self.n_channels, metric='cosine')
            labels = dbscan.fit_predict(X_outliers.cpu().numpy())
            labels_tensor = torch.tensor(labels, device=self.device)
            
            # Filter out the pure noise (DBSCAN labels noise as -1)
            valid_mask = labels_tensor >= 0
            
            if not valid_mask.any():
                print("Shelf full, but only contained scattered noise. Burning shelf.")
            else:
                # Find the largest coherent cluster among the non-noise points
                valid_labels = labels_tensor[valid_mask]
                cluster_counts = torch.bincount(valid_labels)
                dominant_cluster_idx = int(torch.argmax(cluster_counts).item())
                dominant_size = cluster_counts[dominant_cluster_idx].item()
                
                # Require a minimum density to justify birthing a new component
                MIN_PURE_PIXELS = int(self.outlier_update_treshold * 0.75)
                
                if dominant_size >= MIN_PURE_PIXELS:
                    pure_material_mask = (labels_tensor == dominant_cluster_idx)
                    X_pure = X_outliers[pure_material_mask]

                    print(f"Shelf full. DBSCAN isolated {X_pure.shape[0]} pure pixels. Birthing new component.") 

                    cov_matrix = torch.cov(X_pure.T)
                    eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                    eigenvalues = torch.flip(eigenvalues, dims=[0])

                    cumulative_variance = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
                    q_new = torch.searchsorted(cumulative_variance, 0.95).item() + 1
                    print(f"Needed: {q_new} PCs to describe 95% of variance in the setup")
                    
                    if q_new <= self.q_max:
                        with torch.no_grad():
                            # Train the temporary MFA on the pure cluster
                            temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
                            temp_mfa.initialize_parameters(X_pure)
                            temp_mfa.fit(X_pure)

                            # --- NEW: Calculate Sufficient Statistics for the pure cluster ---
                            # Because these pixels are 100% assigned to this component, responsibility = 1
                            N_pure = X_pure.shape[0]
                            new_S0 = torch.tensor([X_pure.shape[0]], dtype=torch.float32, device=self.device)
                            new_S1 = X_pure.sum(dim=0, keepdim=True)            # Shape: (1, D)
                            new_S2 = (X_pure.T @ X_pure).unsqueeze(0) / N_pure  # Shape: (1, D, D)

                            # Pass the parameters AND the statistical mass into the global model
                            self.MFA.add_component(
                                new_mu=temp_mfa.mu.data,
                                new_Lambda=temp_mfa.Lambda.data,
                                new_log_psi=temp_mfa.log_psi.data,
                                new_S0=new_S0,
                                new_S1=new_S1,
                                new_S2=new_S2
                            )

                            # 1. Evaluate the data on the updated MFA (Note: we need log_probs_new here!)
                            log_resp_norm_new_data, _, log_probs_new = self.MFA.e_step(X)
                        
                        # 2. Isolate the responsibilities specifically for the NEW component (the last one)
                        new_comp_idx = self.MFA.K - 1 
                        new_comp_probs = log_resp_norm_new_data[:, new_comp_idx]
                        
                        # 3. Get the top pixels that fit this new component
                        num_to_take = min(2 * self.n_channels, len(X))
                        _, top_indices = torch.topk(new_comp_probs, num_to_take)
                        
                        # 4. Update your trackers and repos using the correct index
                        self.component_pixel_counts = torch.cat([self.component_pixel_counts, torch.tensor([0], device=self.device)])
                        self.local_shelf_counts = torch.cat([self.local_shelf_counts, torch.tensor([0], device=self.device)])
                        self.local_shelves[new_comp_idx] = []
                        self.component_repos[new_comp_idx] = X[top_indices].clone()
                        
                        self.q_limits = torch.cat([self.q_limits, torch.tensor([0.0], device=self.device)])
                        
                        # Compute Q-residuals for the new anchor repository
                        Q_new_repo = self.MFA.compute_q_residuals(self.component_repos[new_comp_idx])[:, new_comp_idx]
                        
                        med_q = torch.median(Q_new_repo)
                        mad_q = torch.median(torch.abs(Q_new_repo - med_q))
                        local_std_q = 1.4826 * mad_q + 1e-6
                        
                        self.q_limits[new_comp_idx] = (med_q + (self.outlier_significance * local_std_q)).item()

            # Burn the shelf (happens regardless of whether a component was birthed or if it was all noise)
            self.num_outliers_on_shelf = 0

        # =====================================================================
        # 3. ADD TO GLOBAL OUTLIER SHELF (If not full yet)
        # =====================================================================
        elif num_new_outliers > 0:
            start_idx = self.num_outliers_on_shelf
            space_left = self.outlier_update_treshold - self.num_outliers_on_shelf
            to_add = min(num_new_outliers, space_left)
            
            end_idx = start_idx + to_add
            self.global_outliers_shelf[start_idx : end_idx] = X[outlier_mask][:to_add]
            self.num_outliers_on_shelf += to_add
                    
        return

    
    def _drift_component(self, k, alpha=0.05):
        """
        Performs a localized, momentum-based M-Step update for a single component.
        Blends the stable 'Anchor' repository with the new 'Drifting' pixels, 
        then recalibrates both the local and global anomaly thresholds.
        """
        # 1. Combine the stable history with the changing present
        drifters = torch.cat(self.local_shelves[k], dim=0)
        anchors = self.component_repos[k]
        
        X_combined = torch.cat([anchors, drifters], dim=0)
        N_comb = X_combined.shape[0]
        
        # 2. Localized M-Step (Responsibility = 1.0 for this specific component)
        mu_proposed = X_combined.mean(dim=0)
        
        diff = X_combined - mu_proposed
        S_k = (diff.T @ diff) / N_comb  # Local Covariance Matrix
        
        try:
            # Weighted PCA for the new Factor Loadings
            vals, vecs = torch.linalg.eigh(S_k)
            idx = torch.argsort(vals, descending=True)
            
            # Keep the global q dimension consistent
            top_vals = torch.clamp(vals[idx[:self.MFA.q]], min=1e-3)
            top_vecs = vecs[:, idx[:self.MFA.q]]
            
            Lambda_proposed = top_vecs * torch.sqrt(top_vals).unsqueeze(0)
            
            # Diagonal Noise Update (Psi)
            recon_cov = Lambda_proposed @ Lambda_proposed.T
            psi_proposed = torch.diagonal(S_k) - torch.diagonal(recon_cov)
            psi_proposed = torch.clamp(psi_proposed, min=1e-3)
            log_psi_proposed = torch.log(psi_proposed)
            
            # Smoothly blend the proposed geometry with the historical geometry
            with torch.no_grad():
                self.MFA.mu.data[k] = (1 - alpha) * self.MFA.mu.data[k] + (alpha * mu_proposed)
                self.MFA.Lambda.data[k] = (1 - alpha) * self.MFA.Lambda.data[k] + (alpha * Lambda_proposed)
                self.MFA.log_psi.data[k] = (1 - alpha) * self.MFA.log_psi.data[k] + (alpha * log_psi_proposed)
                
            print(f"-> Component {k} naturally drifted! (mu, Lambda, Psi updated)")
            
            # Dynamically check actual repo size to prevent IndexError crashes
            actual_repo_size = self.component_repos[k].shape[0]
            num_to_replace = int(actual_repo_size * alpha)
            
            if num_to_replace > 0:
                replace_indices = torch.randperm(actual_repo_size)[:num_to_replace]
                drifter_indices = torch.randperm(drifters.shape[0])[:num_to_replace]
                self.component_repos[k][replace_indices] = drifters[drifter_indices].clone()

            # Calculate the new "normal" boundary for this component
            with torch.no_grad():
                Q_repo = self.MFA.compute_q_residuals(self.component_repos[k])[:, k]
                
                med_q = torch.median(Q_repo)
                mad_q = torch.median(torch.abs(Q_repo - med_q))
                local_std_q = 1.4826 * mad_q + 1e-6
                
                self.q_limits[k] = (med_q + (self.outlier_significance * local_std_q)).item()

        except Exception as e:
            print(f"Warning: Drift failed for component {k} due to numerical instability: {e}")
            
        finally:
            # 7. Burn the local shelf to start collecting new drifters
            # This is in a 'finally' block so the shelf is cleared even if the math above fails
            self.local_shelves[k] = []
