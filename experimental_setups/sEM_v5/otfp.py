import torch
import math
from experimental_setups.sEM_v5.mfa import MFA


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
        
        # --- NEW EMA THRESHOLD TRACKERS ---
        self.global_recon_mean = 0.0
        self.global_recon_var = 1.0
        self.ema_alpha = 0.05 
        self.is_initialized = False
        self.needs_threshold_reset = False
        self.global_threshold = 0.0 
        
        self._run_mfa_setup(init_data)
        
        # Streaming statistics
        self.n_samples_seen = 0
        self.n_model_updates = 0

        # Memory buffers (We only keep the global shelf for birthing new components)
        self.global_outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32) 
        self.num_outliers_on_shelf = 0

        print(f"Finished setup of OTFP-MFA: K={self.MFA.K}, q={self.MFA.q}")

    def _run_mfa_setup(self, X):
        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        with torch.no_grad():
            self.MFA.initialize_parameters(X)
            self.MFA.fit(X) 
            self.MFA.init_sufficient_statistics(X)
            
            # Initial Threshold setup using Reconstruction Error
            errors = self.MFA.compute_reconstruction_error(X)
            best_errors, _ = torch.min(errors, dim=1) # Get the lowest error for each pixel
            
            self.global_recon_mean = torch.mean(best_errors).item()
            self.global_recon_var = torch.var(best_errors, unbiased=False).item()
            
            # Note: For errors, anomalies are ABOVE the threshold, so we ADD the standard deviations
            self.num_sigma = 6.0 
            self.global_threshold = self.global_recon_mean + (self.num_sigma * math.sqrt(self.global_recon_var))
            self.is_initialized = True
            
        print(f"Global reconstruction error threshold initialized: {self.global_threshold:.6f}")
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
            # 1. Find the best reconstruction error for each pixel
            errors = self.MFA.compute_reconstruction_error(X)
            best_errors, best_components = torch.min(errors, dim=1)
            
            # 2. Update EMA Threshold (Burn-in reset if a component was just birthed)
            batch_mean = torch.mean(best_errors)
            batch_var = torch.var(best_errors, unbiased=False)
            
            if not self.is_initialized or self.needs_threshold_reset:
                self.global_recon_mean = batch_mean.item()
                self.global_recon_var = batch_var.item()
                self.is_initialized = True
                self.needs_threshold_reset = False
            else:
                self.global_recon_mean = (1 - self.ema_alpha) * self.global_recon_mean + (self.ema_alpha * batch_mean.item())
                self.global_recon_var = (1 - self.ema_alpha) * self.global_recon_var + (self.ema_alpha * batch_var.item())
            
            # 3. Calculate Threshold (Anomalies have HIGH error)
            self.global_threshold = self.global_recon_mean + (self.num_sigma * math.sqrt(self.global_recon_var))
            
            # 4. Filter Inliers vs Outliers
            outlier_mask = best_errors > self.global_threshold
            inlier_mask = ~outlier_mask
            
            num_new_outliers = outlier_mask.sum().item()

            # --- EM Update for Inliers ---
            if inlier_mask.any():
                X_inliers = X[inlier_mask]
                log_resp_norm, _, _ = self.MFA.e_step(X_inliers)
                self.MFA.stepwise_em_update(X_inliers, log_resp_norm)

        # --- Component Birthing (Cleaned up) ---
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.05, min_samples=2*self.n_channels, metric='cosine')
            labels_tensor = torch.tensor(dbscan.fit_predict(X_outliers.cpu().numpy()), device=self.device)
            
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
                    eigenvalues = torch.flip(torch.linalg.eigvalsh(cov_matrix), dims=[0])
                    cumulative_variance = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
                    q_new = torch.searchsorted(cumulative_variance, 0.95).item() + 1
                    
                    if q_new <= self.q_max:
                        with torch.no_grad():
                            temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
                            temp_mfa.initialize_parameters(X_pure)
                            temp_mfa.fit(X_pure)
                            
                            N_pure = X_pure.shape[0]
                            new_S0 = torch.tensor([1.0], dtype=torch.float32, device=self.device)
                            new_S1 = X_pure.mean(dim=0, keepdim=True)            
                            new_S2 = (X_pure.T @ X_pure).unsqueeze(0) / N_pure  

                            self.MFA.add_component(
                                new_mu=temp_mfa.mu.data,
                                new_Lambda=temp_mfa.Lambda.data,
                                new_log_psi=temp_mfa.log_psi.data,
                                new_S0=new_S0,
                                new_S1=new_S1,
                                new_S2=new_S2
                            )
                            
                            # CRITICAL: Trigger the EMA threshold reset for the next batch!
                            self.needs_threshold_reset = True

            self.num_outliers_on_shelf = 0

        # --- Add new outliers to the shelf ---
        elif num_new_outliers > 0:
            start_idx = self.num_outliers_on_shelf
            space_left = self.outlier_update_treshold - self.num_outliers_on_shelf
            to_add = min(num_new_outliers, space_left)
            
            self.global_outliers_shelf[start_idx : start_idx + to_add] = X[outlier_mask][:to_add]
            self.num_outliers_on_shelf += to_add
                    
        return