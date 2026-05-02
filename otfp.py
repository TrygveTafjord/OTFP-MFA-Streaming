from scipy.stats import chi2
import torch
from mfa import MFA
from bayesian_model_selector import BayesianMFA_Initializer
from sklearn.cluster import DBSCAN

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
            self.MFA.fit(X) 
            self.MFA.init_sufficient_statistics(X)
            
        print(f"Theoretical Chi-Square threshold initialized: {self.chi2_threshold:.2f}")
        return

    def _perform_model_selection(self, data, n_channels, q_max):
        """
        Runs Variational/MAP Bayesian MFA on the initialization data to 
        automatically discover the optimal number of components (K) and factors (q).
        """
        # 1. Define an intentionally large starting assumption
        K_max = 15  # Adjust based on expected maximum initial materials
        
        print(f"Starting Bayesian model selection with K_max={K_max}, q_max={q_max}...")
        
        # Instantiate the Bayesian Initializer
        bayesian_initializer = BayesianMFA_Initializer(
            n_components=K_max, 
            n_channels=n_channels, 
            q_max=q_max, 
            device=self.device
        )
        
        # Fit the model to the initial shelf/batch of data
        bayesian_initializer.fit_with_ard(data)
        
        # Extract the surviving K and q
        with torch.no_grad():
            # Find K: Components whose mixing weights (pi) didn't shrink to zero
            pi_threshold = 1e-3
            pi = torch.exp(bayesian_initializer.log_pi)
            active_components = pi > pi_threshold
            optimal_K = active_components.sum().item()
            
            # Find q: Count surviving factors (alpha < threshold) for the active components
            alpha_threshold = 1e4
            if optimal_K > 0:
                # Get the active factors only for the surviving components
                active_q_per_component = (bayesian_initializer.alpha[active_components] < alpha_threshold).sum(dim=1)
                
                # Taking the max ensures no component is starved of latent capacity.
                optimal_q = active_q_per_component.max().item()
            else:
                optimal_q = 1
        
        # 5. Safety fallbacks in case of aggressive over-pruning
        optimal_K = max(1, optimal_K)
        optimal_q = max(1, optimal_q)
        
        print(f"Model selection complete! Optimal K = {optimal_K}, Optimal q = {optimal_q}")
        
        return optimal_K, optimal_q

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

        # 1. TRACK INLIERS & CATCH LOCAL DRIFTERS
        if inlier_mask.any():
                X_inliers = X[inlier_mask]
                #self._process_inliners(X_inliers)

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
        return
    

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

                    with torch.no_grad():
                        bayesian_spawner = BayesianMFA_Initializer(
                            n_components=1, 
                            n_channels=self.MFA.D, 
                            q_max=global_q, 
                            max_iter=30, 
                            device=self.device
                        )
                        
                        bayesian_spawner.fit_with_ard(X_pure)
                        
                        self.MFA.add_component(
                            X_pure=X_pure,
                            total_samples_seen=self.n_samples_seen,
                            new_mu=bayesian_spawner.mu.data,
                            new_Lambda=bayesian_spawner.Lambda.data, 
                            new_log_psi=bayesian_spawner.log_psi.data
                        )
                        components_birthed += 1
            
            if components_birthed > 0:
                print(f"\nSuccessfully birthed {components_birthed} new components this cycle.")
            else:
                print(f"Clusters found, but none met the minimum size threshold ({MIN_PURE_PIXELS} pixels). Burning shelf.")
            return
