import torch
from mfa import MFA

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
        self.blocks_processed = 0

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
            log_resp_norm, log_likelihood, log_probs = self.MFA.e_step(X)
            
            # Calculate Q-residuals for all training data for our new outlier metric
            Q_all = self.MFA.compute_q_residuals(X)
        
        # 1. --- Q-Residual Thresholds Setup ---
        self.q_thresholds = torch.zeros(self.MFA.K, device=self.device)
        self.num_sigma = 4.0 
        
        # 2. --- Extract Anchors & Set Local Thresholds ---
        self.repo_size = 240
        self.component_repos = {}
        self.local_thresholds = torch.zeros(self.MFA.K, device=self.device)
        
        best_components = torch.argmax(log_resp_norm, dim=1)
        
        for k in range(self.MFA.K):
            mask_k = (best_components == k)
            X_k = X[mask_k]
            
            if len(X_k) > 0:
                # --- A. Set Q-Residual Threshold (For Global Outliers) ---
                q_k = Q_all[mask_k, k]
                median_q = torch.median(q_k)
                mad_q = torch.median(torch.abs(q_k - median_q))
                robust_std_q = 1.4826 * mad_q
                
                # Limit = Median + (Sigma * Robust Std). Above this = Global Outlier
                self.q_thresholds[k] = median_q + (self.num_sigma * robust_std_q)
                
                # --- B. Anchor Extraction ---
                probs_k = log_probs[mask_k, k]
                num_to_take = min(self.repo_size, len(X_k))
                
                # Get the absolute best-fitting pixels (highest log-likelihood)
                top_values, top_indices = torch.topk(probs_k, num_to_take)
                self.component_repos[k] = X_k[top_indices].clone()
                
                # --- C. Set Local Likelihood Threshold (For Drifters) ---
                med_p = torch.median(probs_k)
                mad_p = torch.median(torch.abs(probs_k - med_p))

                local_std = 1.4826 * mad_p
                
                # If a pixel's fit drops below this, it is a shadow/drifter
                self.local_thresholds[k] = (med_p - (3.0 * local_std)).item()
                
            else:
                self.component_repos[k] = torch.empty((0, self.n_channels), device=self.device)
                self.q_thresholds[k] = 1e6 # High fallback so nothing is flagged against an empty component
                self.local_thresholds[k] = -float('inf')

        print(f"Q-residual anomaly thresholds initialized for {self.MFA.K} components.")
        return

    def _perform_model_selection(self, data, n_channels, q_max):
        # Dummy implementation
        K = 4      
        q = 5
        return K, q

    def process_data_block(self, X):
        if self.MFA is None:
            raise RuntimeError("CRITICAL: MFA model was not initialized.")

        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        self.n_samples_seen += X.shape[0]
        
        with torch.no_grad():
            # E-step gives log_probs (geometry) for drifter tracking
            _, _, log_probs = self.MFA.e_step(X)
            
            # Q-residuals gives orthogonal distance for global outliers
            Q_res = self.MFA.compute_q_residuals(X)

        # A pixel is an outlier ONLY if it violates the Q-limit for ALL known components
        outlier_mask = (Q_res > self.q_thresholds).all(dim=1)
        inlier_mask = ~outlier_mask
        
        num_new_outliers = outlier_mask.sum().item()

        # =====================================================================
        # 1. TRACK INLIERS & CATCH LOCAL DRIFTERS (SHADOWS/MIXTURES)
        # =====================================================================
        if inlier_mask.any():
            X_inliers = X[inlier_mask] 
            inlier_probs = log_probs[inlier_mask]
            best_components = torch.argmax(inlier_probs, dim=1)
            
            # Count occurrences of each component and add to our tracker
            counts = torch.bincount(best_components, minlength=self.MFA.K)
            self.component_pixel_counts += counts

            # Catch the drifters
            for k in range(self.MFA.K):
                mask_k = (best_components == k)
                if not mask_k.any():
                    continue
                
                X_k = X_inliers[mask_k]
                geom_fit_k = inlier_probs[mask_k, k]
                
                # Drifters: Good Q-res (shape matches), but bad Log-Likelihood (shadows/shifts)
                drifter_mask = geom_fit_k < self.local_thresholds[k]
                
                if drifter_mask.any():
                    self.local_shelves[k].append(X_k[drifter_mask])
                    
                    # Check if the shelf is full enough to trigger a Momentum M-Step
                    total_drifters = sum([t.shape[0] for t in self.local_shelves[k]])
                    if total_drifters >= 500:
                        self._drift_component(k)

        # =====================================================================
        # 2. GLOBAL OUTLIER SHELF & COMPONENT BIRTHING
        # =====================================================================
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.05, min_samples=int(0.75*self.outlier_update_treshold), metric='cosine')
            labels = dbscan.fit_predict(X_outliers.cpu().numpy())
            labels_tensor = torch.tensor(labels, device=self.device)
            
            valid_mask = labels_tensor >= 0
            
            if not valid_mask.any():
                print("Shelf full, but only contained scattered noise. Burning shelf.")
            else:
                valid_labels = labels_tensor[valid_mask]
                cluster_counts = torch.bincount(valid_labels)
                dominant_cluster_idx = int(torch.argmax(cluster_counts).item())
                dominant_size = cluster_counts[dominant_cluster_idx].item()
                
                MIN_PURE_PIXELS = int(self.outlier_update_treshold * 0.15)
                
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
                            temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
                            temp_mfa.initialize_parameters(X_pure)
                            temp_mfa.fit(X_pure)

                        new_weight = X_pure.shape[0] / max(1, self.n_samples_seen)

                        self.MFA.add_component(
                            new_mu=temp_mfa.mu.data,
                            new_Lambda=temp_mfa.Lambda.data,
                            new_log_psi=temp_mfa.log_psi.data,
                            new_weight=new_weight
                        )

                        new_comp_idx = self.MFA.K - 1 

                        # --- INTEGRATE NEW COMPONENT TRACKERS & THRESHOLDS ---
                        with torch.no_grad():
                            # 1. Set Global Q-Threshold using pure pixels
                            Q_pure = self.MFA.compute_q_residuals(X_pure)
                            q_new_comp = Q_pure[:, new_comp_idx] 
                            
                            median_q_new = torch.median(q_new_comp)
                            mad_q_new = torch.median(torch.abs(q_new_comp - median_q_new))
                            limit_new = median_q_new + (self.num_sigma * 1.4826 * mad_q_new)
                            self.q_thresholds = torch.cat([self.q_thresholds, torch.tensor([limit_new], device=self.device)])
                            
                            # 2. Extract best pure anchors and set Local Likelihood Threshold
                            _, _, log_probs_pure = self.MFA.e_step(X_pure)
                            pure_geom_fits = log_probs_pure[:, new_comp_idx]
                            
                            num_to_take = min(self.repo_size, len(X_pure))
                            top_values, top_indices = torch.topk(pure_geom_fits, num_to_take)
                            
                            med_p = torch.median(pure_geom_fits)
                            mad_p = torch.median(torch.abs(pure_geom_fits - med_p))
                            local_std = 1.4826 * mad_p
                            new_local_thresh = (med_p - (3.0 * local_std)).item()
                            self.local_thresholds = torch.cat([self.local_thresholds, torch.tensor([new_local_thresh], device=self.device)])

                            # 3. Update arrays and repositories
                            self.component_repos[new_comp_idx] = X_pure[top_indices].clone()
                            self.component_pixel_counts = torch.cat([self.component_pixel_counts, torch.tensor([0], device=self.device)])
                            self.local_shelf_counts = torch.cat([self.local_shelf_counts, torch.tensor([0], device=self.device)])
                            self.local_shelves[new_comp_idx] = []

            # Burn the shelf
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
        
        self.blocks_processed += 1
        
        if self.blocks_processed % 100 == 0:
            self.update_global_weights()
            
        return
    

    def update_global_weights(self, decay_factor=0.9):
        """
        Periodically recalibrates the global mixing weights (pi) based on the 
        actual distribution of materials seen recently by the satellite.
        """
        # 1. Decay the old counts so the model adapts to changing landscapes (Exponential Moving Average)
        # We convert to float for the math, then back to long (integers)
        self.component_pixel_counts = (self.component_pixel_counts.float() * decay_factor).long()
        self.local_shelf_counts = (self.local_shelf_counts.float() * decay_factor).long()
        
        # 2. Combine all known pixels assigned to each component
        total_counts = self.component_pixel_counts + self.local_shelf_counts
        
        # 3. Apply Laplace Smoothing to prevent log(0) for new/rare components
        smoothed_counts = total_counts + 1.0 
        
        # 4. Calculate the new linear probabilities
        new_pi = smoothed_counts / smoothed_counts.sum()
        
        # 5. Safely inject the new weights into the MFA model in log-space
        with torch.no_grad():
            self.MFA.log_pi.data = torch.log(new_pi)
    
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
            top_vals = torch.clamp(vals[idx[:self.MFA.q]], min=1e-6)
            top_vecs = vecs[:, idx[:self.MFA.q]]
            
            Lambda_proposed = top_vecs * torch.sqrt(top_vals).unsqueeze(0)
            
            # Diagonal Noise Update (Psi)
            recon_cov = Lambda_proposed @ Lambda_proposed.T
            psi_proposed = torch.diagonal(S_k) - torch.diagonal(recon_cov)
            psi_proposed = torch.clamp(psi_proposed, min=1e-6)
            log_psi_proposed = torch.log(psi_proposed)
            
            # 3. --- APPLY MOMENTUM ---
            # Smoothly blend the proposed geometry with the historical geometry
            with torch.no_grad():
                self.MFA.mu.data[k] = (1 - alpha) * self.MFA.mu.data[k] + (alpha * mu_proposed)
                self.MFA.Lambda.data[k] = (1 - alpha) * self.MFA.Lambda.data[k] + (alpha * Lambda_proposed)
                self.MFA.log_psi.data[k] = (1 - alpha) * self.MFA.log_psi.data[k] + (alpha * log_psi_proposed)
                
            print(f"-> Component {k} naturally drifted! (mu, Lambda, Psi updated)")
            
            # 4. --- UPDATE ANCHOR PIXELS (Reservoir Sampling) ---
            # Dynamically check actual repo size to prevent IndexError crashes
            actual_repo_size = self.component_repos[k].shape[0]
            num_to_replace = int(actual_repo_size * alpha)
            
            if num_to_replace > 0:
                replace_indices = torch.randperm(actual_repo_size)[:num_to_replace]
                drifter_indices = torch.randperm(drifters.shape[0])[:num_to_replace]
                self.component_repos[k][replace_indices] = drifters[drifter_indices].clone()

            # 5. --- RECALIBRATE LOCAL THRESHOLD ---
            # Calculate the new "normal" boundary for this component
            with torch.no_grad():
                new_geom_fits = self.MFA.compute_component_log_likelihoods(self.component_repos[k])[:, k]
                
                med_p = torch.median(new_geom_fits)
                mad_p = torch.median(torch.abs(new_geom_fits - med_p))
                local_std = 1.4826 * mad_p
                
                self.local_thresholds[k] = (med_p - (3.0 * local_std)).item()

            # 6. --- RECALIBRATE Q-RESIDUAL THRESHOLD ---
            # The component's shape shifted, so its Q-residual limit must be updated.
            with torch.no_grad():
                # Re-evaluate the updated anchors against the updated MFA model
                Q_anchors = self.MFA.compute_q_residuals(self.component_repos[k])
                q_k = Q_anchors[:, k]
                
                median_q = torch.median(q_k)
                mad_q = torch.median(torch.abs(q_k - median_q))
                robust_std_q = 1.4826 * mad_q
                
                # Limit = Median + (Sigma * Robust Std)
                self.q_thresholds[k] = median_q + (self.num_sigma * robust_std_q)

        except Exception as e:
            print(f"Warning: Drift failed for component {k} due to numerical instability: {e}")
            
        finally:
            # 7. Burn the local shelf to start collecting new drifters
            # This is in a 'finally' block so the shelf is cleared even if the math above fails
            self.local_shelves[k] = []