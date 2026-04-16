from scipy.stats import chi2
import torch
from mfa import MFA

class MFA_OTFP:
    def __init__(self, init_data: torch.Tensor, n_channels: int, device: str, outlier_update_treshold: int, L2_normalization: bool = True, q_max: int = 5):
        # System hyperparameters
        self.device = device
        self.n_channels = n_channels
        self.q_max = q_max
        self.outlier_update_treshold = outlier_update_treshold
        self.L2_normalization = L2_normalization

        # MFA model-state 
        K, q = self._perform_model_selection(data=init_data, n_channels=n_channels, q_max=q_max)
        self.MFA = MFA(n_components=K, n_channels=n_channels, n_factors=q).to(self.device)

        # Use a single global threshold
        self.chi2_threshold = float(chi2.ppf(0.9999, df=self.n_channels))
        
        self._run_mfa_setup(init_data)
        
        # Streaming statistics
        self.n_samples_seen = 0
        self.n_model_updates = 0

        # Memory buffers
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
            
        print(f"Theoretical Chi-Square threshold initialized: {self.chi2_threshold:.2f}")
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
            _, _, _, mahalanobis_dists = self.MFA.e_step(X)

        min_mahalanobis, _ = torch.min(mahalanobis_dists, dim=1)
        outlier_mask = min_mahalanobis > self.chi2_threshold

        inlier_mask = ~outlier_mask
        
        num_new_outliers = outlier_mask.sum().item()

        # =====================================================================
        # 1. TRACK INLIERS & CATCH LOCAL DRIFTERS (SHADOWS/MIXTURES)
        # =====================================================================
        if inlier_mask.any():
            with torch.no_grad():
                X_inliers = X[inlier_mask]

                # Use the E-step to get the normalized responsibilities for the inliers
                log_resp_norm, _, _, _ = self.MFA.e_step(X_inliers)

                # Stream the block directly into the Stepwise EM updater
                self.MFA.stepwise_em_update(X_inliers, log_resp_norm)
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
                            
                            # Calculate Sufficient 
                            N_pure = X_pure.shape[0]
                            
                            # Because these pixels are 100% assigned to this new component, 
                            # the average responsibility per pixel is exactly 1.0
                            new_S0 = torch.tensor([1.0], dtype=torch.float32, device=self.device)
                            
                            # Mean instead of sum
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

                            # 1. Evaluate the data on the updated MFA (Note: we need log_probs_new here!)
                            log_resp_norm_new_data, _, log_probs_new, _ = self.MFA.e_step(X)

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

