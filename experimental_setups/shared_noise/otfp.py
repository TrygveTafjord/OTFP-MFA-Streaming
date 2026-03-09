import torch
from experimental_setups.shared_noise.mfa import MFA

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

        # Use a single global threshold
        self.global_threshold = 0.0 
        
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
        
        # Global Threshold Setup (Median + MAD)
        median_ll = torch.median(log_likelihood)
        mad_ll = torch.median(torch.abs(log_likelihood - median_ll))
        robust_std_ll = 1.4826 * mad_ll

        
        
        self.num_sigma = 6.0 
        self.global_threshold = (median_ll - (self.num_sigma * robust_std_ll)).item()
        
        # 2. --- Extract Anchors & Set Local Thresholds ---
        self.repo_size = 360
        self.component_repos = {}
        self.local_thresholds = torch.zeros(self.MFA.K, device=self.device)
        
        best_components = torch.argmax(log_resp_norm, dim=1)
        
        for k in range(self.MFA.K):
            mask_k = (best_components == k)
            X_k = X[mask_k]
            
            if len(X_k) > 0:
                probs_k = log_probs[mask_k, k]
                num_to_take = min(self.repo_size, len(X_k))
                
                # Get the absolute best-fitting pixels
                top_values, top_indices = torch.topk(probs_k, num_to_take)
                self.component_repos[k] = X_k[top_indices].clone()
                
                # Set the local 3-sigma threshold for THIS specific component
                med_p = torch.median(top_values)
                mad_p = torch.median(torch.abs(top_values - med_p))
                local_std = 1.4826 * mad_p
                
                # If a pixel's fit drops below this, it is a shadow/drifter
                self.local_thresholds[k] = (med_p - (3.0 * local_std)).item()
                
            else:
                self.component_repos[k] = torch.empty((0, self.n_channels), device=self.device)
                self.local_thresholds[k] = -float('inf')

        print(f"Global anomaly threshold initialized: {self.global_threshold:.2f}")
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
        
        with torch.no_grad():
            # We need log_probs back to figure out which component the inliers belong to
            _, log_likelihood, log_probs = self.MFA.e_step(X)

        outlier_mask = log_likelihood < self.global_threshold
        inlier_mask = ~outlier_mask
        
        num_new_outliers = outlier_mask.sum().item()

        # =====================================================================
        # 1. TRACK INLIERS & CATCH LOCAL DRIFTERS (SHADOWS/MIXTURES)
        # =====================================================================
        if inlier_mask.any():
            X_inliers = X[inlier_mask]  # We need the actual pixel data to save drifters
            inlier_probs = log_probs[inlier_mask]
            best_components = torch.argmax(inlier_probs, dim=1)
            
            # Count occurrences of each component and add to our tracker
            counts = torch.bincount(best_components, minlength=self.MFA.K)
            self.component_pixel_counts += counts

            # --- STEP 2: CATCH THE DRIFTERS IN THE STREAM ---
            for k in range(self.MFA.K):
                mask_k = (best_components == k)
                if not mask_k.any():
                    continue
                
                X_k = X_inliers[mask_k]
                geom_fit_k = inlier_probs[mask_k, k]
                
                # Identify the pixels that are "drifting" (shadows, mixtures, seasonal change)
                drifter_mask = geom_fit_k < self.local_thresholds[k]
                
                if drifter_mask.any():
                    X_drifters = X_k[drifter_mask]
                    self.local_shelves[k].append(X_drifters)
                    
                    # Check if the shelf is full enough to trigger a Momentum M-Step
                    total_drifters = sum([t.shape[0] for t in self.local_shelves[k]])
                    if total_drifters >= 500:
                        self._drift_component(k)
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
                            temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
                            temp_mfa.initialize_parameters(X_pure)
                            temp_mfa.fit(X_pure)

                            new_weight = X_pure.shape[0] / max(1, self.n_samples_seen)

                            self.MFA.add_component(
                                new_mu=temp_mfa.mu.data,
                                new_Lambda=temp_mfa.Lambda.data,
                                new_weight=new_weight
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
                        
                        # --- FIX: INITIALIZE THE LOCAL THRESHOLD FOR THE NEW COMPONENT ---
                        self.local_thresholds = torch.cat([self.local_thresholds, torch.tensor([0.0], device=self.device)])
                        new_geom_fits = log_probs_new[top_indices, new_comp_idx]
                        med_p = torch.median(new_geom_fits)
                        mad_p = torch.median(torch.abs(new_geom_fits - med_p))
                        local_std = 1.4826 * mad_p
                        self.local_thresholds[new_comp_idx] = (med_p - (3.0 * local_std)).item()
                        # -----------------------------------------------------------------

                        # 5. Concatenate all anchor pixels across all components
                        repo_tensors = [self.component_repos[i] for i in range(self.MFA.K)]
                        total_dataset = torch.cat(repo_tensors, dim=0)
                        
                        # 6. Calculate the new global threshold based on the combined anchors
                        with torch.no_grad():
                            log_resp_norm, log_likelihood, log_probs = self.MFA.e_step(total_dataset)
                        
                        median_ll = torch.median(log_likelihood)
                        mad_ll = torch.median(torch.abs(log_likelihood - median_ll))
                        robust_std_ll = 1.4826 * mad_ll
                        
                        self.num_sigma = 6.0 
                        self.global_threshold = (median_ll - (self.num_sigma * robust_std_ll)).item()

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
            
            # NOTE: We no longer calculate or update psi_proposed here!
            # The sensor noise is a global assumption, not a local one.
            
            # Smoothly blend the proposed geometry with the historical geometry
            with torch.no_grad():
                self.MFA.mu.data[k] = (1 - alpha) * self.MFA.mu.data[k] + (alpha * mu_proposed)
                self.MFA.Lambda.data[k] = (1 - alpha) * self.MFA.Lambda.data[k] + (alpha * Lambda_proposed)
                # DELETED: self.MFA.log_psi.data[k] = ... 
                
            print(f"-> Component {k} naturally drifted! (mu and Lambda updated)")
            
            # Dynamically check actual repo size to prevent IndexError crashes
            actual_repo_size = self.component_repos[k].shape[0]
            num_to_replace = int(actual_repo_size * alpha)
            
            if num_to_replace > 0:
                replace_indices = torch.randperm(actual_repo_size)[:num_to_replace]
                drifter_indices = torch.randperm(drifters.shape[0])[:num_to_replace]
                self.component_repos[k][replace_indices] = drifters[drifter_indices].clone()

            # Calculate the new "normal" boundary for this component
            with torch.no_grad():
                new_geom_fits = self.MFA.compute_component_log_likelihoods(self.component_repos[k])[:, k]
                
                med_p = torch.median(new_geom_fits)
                mad_p = torch.median(torch.abs(new_geom_fits - med_p))
                local_std = 1.4826 * mad_p
                
                self.local_thresholds[k] = (med_p - (3.0 * local_std)).item()

            # The world just shifted, evaluate combined anchors to find the new global baseline
            with torch.no_grad():
                repo_tensors = [self.component_repos[i] for i in range(self.MFA.K)]
                total_dataset = torch.cat(repo_tensors, dim=0)
                
                _, global_ll, _ = self.MFA.e_step(total_dataset)
                
                median_ll = torch.median(global_ll)
                mad_ll = torch.median(torch.abs(global_ll - median_ll))
                robust_std_ll = 1.4826 * mad_ll
                
                self.global_threshold = (median_ll - (self.num_sigma * robust_std_ll)).item()

        except Exception as e:
            print(f"Warning: Drift failed for component {k} due to numerical instability: {e}")
            
        finally:
            # 7. Burn the local shelf to start collecting new drifters
            self.local_shelves[k] = []