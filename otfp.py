from scipy.stats import chi2
import torch
from mfa import MFA
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
            n_components=0, 
            n_channels=n_channels, 
            n_factors=q_max,
            device=device
        ).to(device)

        self.MFA_fitted = False

        # Use a single global threshold
        self.chi2_threshold = float(chi2.ppf(0.9999, df=self.n_channels))
                
        # Streaming statistics
        self.n_samples_seen = 0
        self.n_model_updates = 0

        # Memory buffers
        self.global_outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32) 
        self.num_outliers_on_shelf = 0

        return
    
    def fit(self, X):
        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)        
        self._birth_new_components(X)
        return

    def process_data_block(self, X, stepwise_updates: bool = True):

        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        self.n_samples_seen += X.shape[0]

        if not self.MFA_fitted:
            min_mahalanobis = torch.full_like(torch.zeros(X.shape[0]), fill_value=(self.chi2_threshold + 1.0))  # Mark all as outliers if model isn't fitted yet
            assignments = torch.full_like(torch.zeros(X.shape[0], dtype=torch.long), fill_value=-1)  # No valid assignments
            log_resp_norm = None  # No responsibilities to return yet
            assigned_mahalanobis = min_mahalanobis  
            
        else: 
        
            with torch.no_grad():
                # Unpack log_resp_norm to find the most probable component (MAP assignment)
                log_resp_norm, _, _, mahalanobis_dists = self.MFA.e_step(X)

            assignments = torch.argmax(log_resp_norm, dim=1)
            assigned_mahalanobis = mahalanobis_dists.gather(1, assignments.unsqueeze(1)).squeeze(1)
            min_mahalanobis, _ = torch.min(mahalanobis_dists, dim=1)

        outlier_mask = min_mahalanobis > self.chi2_threshold
        assignments[outlier_mask] = -1  # Mark outliers with a special assignment value (e.g., -1)
        inlier_mask = ~outlier_mask
        num_new_outliers = outlier_mask.sum().item()

        # 1. TRACK INLIERS & CATCH LOCAL DRIFTERS
        if inlier_mask.any() and stepwise_updates:
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
            if self.L2_normalization:
                metric = 'cosine'
                dbscan_eps = 0.001
            else:
                metric = 'euclidean'
                dbscan_eps = 50.0

            dbscan = DBSCAN(eps=dbscan_eps, min_samples=self.n_channels, metric=metric)
            labels = dbscan.fit_predict(X_outliers.cpu().numpy())
            labels_tensor = torch.tensor(labels, device=self.device)
            
            # Filter out the pure noise (DBSCAN labels noise as -1)
            valid_mask = labels_tensor >= 0
            
            if not valid_mask.any():
                # print("Shelf full, but only contained scattered noise. Burning shelf.")
                return
            
            # Find all unique clusters and their sizes
            valid_labels = labels_tensor[valid_mask]
            unique_clusters, cluster_counts = torch.unique(valid_labels, return_counts=True)
            
            num_valid_clusters = len(unique_clusters)

            # 2. Gather all samples that belong to ANY valid cluster
            # (Since we filtered out -1, all remaining unique_clusters are valid)
            pure_materials_mask = torch.isin(labels_tensor, unique_clusters)
            X_all_valid = X_outliers[pure_materials_mask]
            
            global_q = self.MFA.q

            # 3. Train an MFA on all the valid cluster-samples together
            cluster_model = MFA(
                n_components=num_valid_clusters,
                n_channels=self.MFA.D,
                n_factors=global_q,
                device=self.device
            )
            
            cluster_model.fit(X_all_valid, n_init=5)
            
            # Re-assign the points to the newly fitted MFA components
            with torch.no_grad():
                _, _, _, mahalanobis = cluster_model.e_step(X_all_valid)
                new_assignments = mahalanobis.argmin(dim=1)
                
                # Only pass the components that actually have data assigned to them
                self.MFA.add_components(
                    X_valid=X_all_valid,
                    assignments=new_assignments, 
                    total_samples_seen=self.n_samples_seen,
                    new_mu=cluster_model.mu.data,
                    new_Lambda=cluster_model.Lambda.data,
                    new_log_psi=cluster_model.log_psi.data
                )

                self.MFA_fitted = True
            
            return
    