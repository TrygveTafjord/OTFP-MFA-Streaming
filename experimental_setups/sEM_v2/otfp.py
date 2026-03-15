import torch
from experimental_setups.sEM_v2.mfa import MFA
from sklearn.cluster import DBSCAN

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
        
        # Local boundaries for each component
        self.local_thresholds = 0
        
        # Streaming statistics
        self.n_samples_seen = 0
        self.n_model_updates = 0

        # Memory buffers (Strictly bounded memory, NO anchor repos)
        self.global_outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32) 
        self.num_outliers_on_shelf = 0

        # lOCAL UPDATE TRESHOLD STATISTICS
        self.local_ll_medians = torch.zeros(self.MFA.K, device=self.device)
        self.local_ll_mads = torch.zeros(self.MFA.K, device=self.device)
        self.local_pixel_counts = torch.zeros(self.MFA.K, device=self.device)

        self._run_mfa_setup(init_data)
        
        print(f"Finished setup of OTFP-MFA: K={self.MFA.K}, q={self.MFA.q}")

    def _run_mfa_setup(self, X):
        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        with torch.no_grad():
            self.MFA.initialize_parameters(X)
            self.MFA.fit(X) 
            self.MFA.init_sufficient_statistics(X)
            _, log_likelihood, _ = self.MFA.e_step(X)
        
        # Global Threshold Setup (Median + MAD)
        median_ll = torch.median(log_likelihood)
        mad_ll = torch.median(torch.abs(log_likelihood - median_ll))
        robust_std_ll = 1.4826 * mad_ll
        
        self.num_sigma = 6.0 
        self.global_threshold = (median_ll - (self.num_sigma * robust_std_ll)).item()
        
        print(f"Global anomaly threshold initialized: {self.global_threshold:.2f}")
        return

    def _perform_model_selection(self, data, n_channels, q_max):
        # Dummy implementation - insert your BIC grid search here
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
            # log_probs shape: (N, K)
            log_resp_norm, log_likelihood, _ = self.MFA.e_step(X)
            batch_min = log_likelihood.min().item()
            batch_max = log_likelihood.max().item()
            batch_med = torch.median(log_likelihood).item()
            
            print(f"[DEBUG BATCH] LL Range: [{batch_min:.2f} to {batch_max:.2f}] | Median: {batch_med:.2f} | Global Threshold: {self.global_threshold:.2f}")
            # ----------------------------------

        # =====================================================================
        # 1. LOCAL BOUNDING BOX CHECK
        # =====================================================================
        # A pixel is an inlier if its log_prob is greater than the threshold 
        # for AT LEAST ONE component.
        # log_probs shape: (N, K), local_thresholds shape: (K,)
        inlier_mask = log_likelihood > self.global_threshold 
        outlier_mask = ~inlier_mask
        
        num_new_outliers = outlier_mask.sum().item()

        # =====================================================================
        # 2. TRACK INLIERS via STEPWISE EM
        # =====================================================================
        if inlier_mask.any():
            X_inliers = X[inlier_mask]
            log_resp_inliers = log_resp_norm[inlier_mask]
            
            with torch.no_grad():
                # Stream the block directly into the Stepwise EM updater
                self.MFA.stepwise_em_update(X_inliers, log_resp_inliers)
                
        # =====================================================================
        # 3. GLOBAL OUTLIER SHELF & COMPONENT BIRTHING
        # =====================================================================
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            # Use DBSCAN to isolate true signals from random noise
            dbscan = DBSCAN(eps=0.05, min_samples=int(0.3*(num_new_outliers + self.num_outliers_on_shelf)), metric='cosine')
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
                    
                    if q_new <= self.q_max:
                        with torch.no_grad():
                            # Train the temporary MFA on the pure cluster
                            temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
                            temp_mfa.initialize_parameters(X_pure)
                            temp_mfa.fit(X_pure)

                            # In otfp.py, when passing stats to add_component:
                            N_pure = X_pure.shape[0]

                            # Average scale (N_pure / N_pure = 1.0 for S0)
                            new_S0 = torch.tensor([1.0], dtype=torch.float32, device=self.device) 
                            new_S1 = X_pure.mean(dim=0, keepdim=True)            
                            new_S2 = ((X_pure.T @ X_pure) / N_pure).unsqueeze(0)  

                            self.MFA.add_component(
                                new_mu=temp_mfa.mu.data,
                                new_Lambda=temp_mfa.Lambda.data,
                                new_log_psi=temp_mfa.log_psi.data,
                                new_S0=new_S0,
                                new_S1=new_S1,
                                new_S2=new_S2
                            )

                            with torch.no_grad():
                                combined_X = torch.cat([X, X_pure], dim=0)
                                _, updated_ll, _ = self.MFA.e_step(combined_X)
                                
                                batch_median = torch.median(updated_ll)
                                batch_mad = torch.median(torch.abs(updated_ll - batch_median))
                                batch_std = 1.4826 * batch_mad
                                
                                # Calculate what this specific block thinks the threshold should be
                                batch_threshold = (batch_median - (self.num_sigma * batch_std)).item()
                                
                                # Blend historical strictness with new capacity (e.g., 50/50 split)
                                # A higher alpha (e.g., 0.8) heavily favors the new batch
                                # A lower alpha (e.g., 0.2) heavily favors history
                                # If the new threshold needs to drop (new capacity added), let it adapt over ~10 batches
                                if batch_threshold < self.global_threshold:
                                    alpha_thresh = 0.05  # 5% new reality, 95% history
                                # If it needs to rise, make it extremely stubborn so noise doesn't lock out valid data
                                else:
                                    alpha_thresh = 0.001 # 0.1% new reality, 99.9% history
                                
                                self.global_threshold = ((1.0 - alpha_thresh) * self.global_threshold) + (alpha_thresh * batch_threshold)
                            
            # Burn the shelf
            self.num_outliers_on_shelf = 0

        # =====================================================================
        # 4. ADD TO GLOBAL OUTLIER SHELF (If not full yet)
        # =====================================================================
        elif num_new_outliers > 0:
            start_idx = self.num_outliers_on_shelf
            space_left = self.outlier_update_treshold - self.num_outliers_on_shelf
            to_add = min(num_new_outliers, space_left)
            
            end_idx = start_idx + to_add
            self.global_outliers_shelf[start_idx : end_idx] = X[outlier_mask][:to_add]
            self.num_outliers_on_shelf += to_add
                    
        return