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
        self.local_thresholds = torch.zeros(self.MFA.K, device=self.device)
        
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
            
            # We only need the raw geometric fits for thresholding
            _, _, log_probs = self.MFA.e_step(X)
        
        # --- Extract Local Thresholds ---
        # Assign each setup pixel to its best-fitting component
        best_components = torch.argmax(log_probs, dim=1)
        
        for k in range(self.MFA.K):
            mask_k = (best_components == k)
            probs_k = log_probs[mask_k, k]
            
            if len(probs_k) > 0:
                # Set the local sigma threshold for THIS specific component
                med_p = torch.median(probs_k)
                mad_p = torch.median(torch.abs(probs_k - med_p))
                local_std = 1.4826 * mad_p
                
                # If a pixel's fit to component k drops below this, it is an outlier to k
                self.local_thresholds[k] = (med_p - (self.outlier_significance * local_std)).item()
            else:
                self.local_thresholds[k] = -float('inf')

        print(f"Initial local thresholds established for {self.MFA.K} components.")
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
            log_resp_norm, _, log_probs = self.MFA.e_step(X)

        # =====================================================================
        # 1. LOCAL BOUNDING BOX CHECK
        # =====================================================================
        # A pixel is an inlier if its log_prob is greater than the threshold 
        # for AT LEAST ONE component.
        # log_probs shape: (N, K), local_thresholds shape: (K,)
        is_inlier_matrix = log_probs > self.local_thresholds.unsqueeze(0) 
        inlier_mask = is_inlier_matrix.any(dim=1) # (N,)
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
                
                # --- NEW DYNAMIC THRESHOLD LOOP ---
                # 1. Re-evaluate log-likelihoods of inliers against the UPDATED model
                # Shape: (N_inliers, K)
                updated_ll = self.MFA.compute_component_log_likelihoods(X_inliers)
                
                # 2. Find hard assignments to know which pixel belongs to which component
                best_components = torch.argmax(updated_ll, dim=1)
                
                # 3. Loop through each component to update its specific threshold
                # 3. Loop through each component to update its specific threshold
                for k in range(self.MFA.K):
                    # Isolate pixels that belong to component k
                    mask_k = (best_components == k)
                    N_batch = mask_k.sum().float() # Number of pixels in this specific batch
                    
                    # If no pixels in this batch belong to component k, skip its threshold update
                    if N_batch == 0:
                        continue 
                    
                    # Extract the log-likelihoods of ONLY component k's pixels
                    new_ll_k = updated_ll[mask_k, k]
                    
                    # Calculate Median and MAD for THIS batch for THIS component
                    batch_med = torch.median(new_ll_k)
                    batch_mad = torch.median(torch.abs(new_ll_k - batch_med))
                    
                    # --- THE SAMPLE-WEIGHTED sEM UPDATE ---
                    # 1. Update the total number of pixels this component has evaluated
                    self.local_pixel_counts[k] += N_batch
                    
                    # 2. Calculate rho: proportion of new data vs total historical data
                    rho = (N_batch / self.local_pixel_counts[k]).item()
                    
                    # 3. Clamp rho to maintain long-term plasticity 
                    # (Prevents rho from reaching absolute 0 when counts get into the millions)
                    rho = max(rho, 0.01) 
                    
                    # 4. Update the historical sufficient statistics of the log-likelihood
                    # (If this is the first batch, rho=1.0, which perfectly overwrites the 0.0 initial state)
                    self.local_ll_medians[k] = (1 - rho) * self.local_ll_medians[k] + (rho * batch_med)
                    self.local_ll_mads[k] = (1 - rho) * self.local_ll_mads[k] + (rho * batch_mad)
                        
                    # 5. Derive the new threshold from the robust historical statistics
                    historical_std = 1.4826 * self.local_ll_mads[k] + 1e-4
                    self.local_thresholds[k] = self.local_ll_medians[k] - (self.outlier_significance * historical_std)
                    
                    # Optional: Print for debugging
                    # print(f"Comp {k} Threshold updated. Added {int(N_batch.item())} px. New tau: {self.local_thresholds[k].item():.2f} (rho={rho:.3f})")

        # =====================================================================
        # 3. GLOBAL OUTLIER SHELF & COMPONENT BIRTHING
        # =====================================================================
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            # Use DBSCAN to isolate true signals from random noise
            dbscan = DBSCAN(eps=0.05, min_samples=int(0.75*(num_new_outliers + self.num_outliers_on_shelf)), metric='cosine')
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

                            # --- INITIALIZE THE LOCAL THRESHOLD FOR THE NEW COMPONENT ---
                            # Evaluate the pure pixels on the temporary geometry to find their baseline fit
                            new_geom_fits = temp_mfa.compute_component_log_likelihoods(X_pure)[:, 0]
                            med_p = torch.median(new_geom_fits)
                            mad_p = torch.median(torch.abs(new_geom_fits - med_p))
                            local_std = 1.4826 * mad_p + 1e-4
                            
                            new_tau = torch.tensor([(med_p - (self.outlier_significance * local_std)).item()], device=self.device)
                            
                            # Append the new threshold to the tracker
                            self.local_thresholds = torch.cat([self.local_thresholds, new_tau])
                            
                            # CRITICAL FIX: Append to the historical sEM trackers
                            self.local_ll_medians = torch.cat([self.local_ll_medians, torch.tensor([med_p.item()], device=self.device)])
                            self.local_ll_mads = torch.cat([self.local_ll_mads, torch.tensor([mad_p.item()], device=self.device)])
                            
                            # THE NEW FIX: Register the number of pixels used to birth this component
                            self.local_pixel_counts = torch.cat([self.local_pixel_counts, torch.tensor([float(N_pure)], device=self.device)])
                            
                            print(f"Component birthed from {N_pure} pixels. Local threshold set to {new_tau.item():.2f}")

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