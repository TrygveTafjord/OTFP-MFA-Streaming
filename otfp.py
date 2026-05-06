from scipy.stats import chi2
import torch
from mfa import MFA
from bayesian_model_selector import BayesianMFA_Initializer
from sklearn.cluster import DBSCAN

class MFA_OTFP:
    def __init__(self, n_channels: int, device: str, outlier_update_treshold: int, L2_normalization: bool = True, q_max: int = 8):
        # System hyperparameters
        self.device = device
        self.n_channels = n_channels
        self.q_max = q_max
        self.outlier_update_treshold = outlier_update_treshold
        self.L2_normalization = L2_normalization

        # MFA model-state 
        self.MFA = MFA(
            n_components=1, 
            n_channels=n_channels, 
            n_factors=q_max,
            device=device
        ).to(device)

        # Use a single global threshold
        self.chi2_threshold = float(chi2.ppf(0.9999, df=self.n_channels))
                
        # Streaming statistics
        self.n_samples_seen = 0
        self.n_model_updates = 0

        # Memory buffers
        self.global_outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32) 
        self.num_outliers_on_shelf = 0

        return


    def process_data_block(self, X):

        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        self.n_samples_seen += X.shape[0]
        
        with torch.no_grad():
            # Unpack log_resp_norm to find the most probable component (MAP assignment)
            log_resp_norm, _, _, mahalanobis_dists = self.MFA.e_step(X)
        
        # 1. MAP Assignment: argmax over the normalized log responsibilities
        assignments = torch.argmax(log_resp_norm, dim=1)
        
        # 2. Distance Extraction: Use gather to pick the distance of the assigned component
        assigned_mahalanobis = mahalanobis_dists.gather(1, assignments.unsqueeze(1)).squeeze(1)

        # Original outlier masking logic using absolute geometric distances
        min_mahalanobis, _ = torch.min(mahalanobis_dists, dim=1)
        outlier_mask = min_mahalanobis > self.chi2_threshold
        inlier_mask = ~outlier_mask
        num_new_outliers = outlier_mask.sum().item()

        # 1. TRACK INLIERS & CATCH LOCAL DRIFTERS
        if inlier_mask.any():
            X_inliers = X[inlier_mask]
            self._process_inliners(X_inliers)

        # 2. GLOBAL OUTLIER SHELF & COMPONENT BIRTHING
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            self._birth_new_components(X_outliers)
            
            self.num_outliers_on_shelf = 0

        # 3. ADD TO GLOBAL OUTLIER SHELF
        elif num_new_outliers > 0:

            start_idx = self.num_outliers_on_shelf            
            end_idx = start_idx + num_new_outliers
            
            self.global_outliers_shelf[start_idx : end_idx] = X[outlier_mask][:num_new_outliers]
            self.num_outliers_on_shelf += num_new_outliers            

        # Return the MAP assignments, the geometric distances, and the statistical confidences
        return assignments, assigned_mahalanobis, log_resp_norm
    

    def _process_inliners(self, X_inliers):
        with torch.no_grad():
            log_resp_norm, _, _, _ = self.MFA.e_step(X_inliers)
            self.MFA.stepwise_em_update(X_inliers, log_resp_norm)
        return
    

    def _birth_new_components(self, X_outliers):
            
            dbscan = DBSCAN(eps=0.05, min_samples=2*self.n_channels, metric='cosine')
            labels = dbscan.fit_predict(X_outliers.cpu().numpy())
            labels_tensor = torch.tensor(labels, device=self.device)
            
            # Filter out the pure noise (DBSCAN labels noise as -1)
            valid_mask = labels_tensor >= 0
            
            if not valid_mask.any():
                print("Shelf full, but only contained scattered noise. Burning shelf.")
                return
            
            # Find all unique clusters and their sizes
            valid_labels = labels_tensor[valid_mask]
            unique_clusters, cluster_counts = torch.unique(valid_labels, return_counts=True)
            
            MIN_PURE_PIXELS = 2 * self.n_channels
            components_birthed = 0
            
            # Loop over clusters found by DBSCAN and check if they meet the minimum size threshold to be considered a pure material cluster
            for cluster_idx, size in zip(unique_clusters, cluster_counts):
                if size.item() >= MIN_PURE_PIXELS:
                    pure_material_mask = (labels_tensor == cluster_idx)
                    X_pure = X_outliers[pure_material_mask]
                    global_q = self.MFA.q

                    cluster_model = MFA(
                        n_components=1,
                        n_channels=self.MFA.D,
                        n_factors=global_q,
                        device=self.device
                    )

                    cluster_model.fit(X_pure, n_init=5)
                        
                    self.MFA.add_component(
                        X_pure=X_pure,
                        total_samples_seen=self.n_samples_seen,
                        new_mu=cluster_model.mu.data,
                        new_Lambda=cluster_model.Lambda.data, 
                        new_log_psi=cluster_model.log_psi.data
                    )
                    components_birthed += 1
            
            if components_birthed > 0:
                print(f"\nSuccessfully birthed {components_birthed} new components this cycle.")
            else:
                print(f"Clusters found, but none met the minimum size threshold ({MIN_PURE_PIXELS} pixels). Burning shelf.")
            return