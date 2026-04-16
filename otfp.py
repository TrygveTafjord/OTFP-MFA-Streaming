from scipy.stats import chi2
import torch
from mfa import MFA
from bayesian_model_selector import BayesianMFA_Initializer

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
            #self.MFA.initialize_parameters(X)
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
        
        # 2. Instantiate the Bayesian Initializer
        # (Assuming BayesianMFA_Initializer is imported from mfa.py)
        bayesian_initializer = BayesianMFA_Initializer(
            n_components=K_max, 
            n_channels=n_channels, 
            q_max=q_max, 
            device=self.device
        )
        
        # 3. Fit the model to the initial shelf/batch of data
        bayesian_initializer.fit_with_ard(data)
        
        # 4. Extract the surviving K and q
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
                
                # Your standard MFA class expects a single global q. 
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
            dbscan = DBSCAN(eps=0.05, min_samples=2*self.n_channels, metric='cosine')
            labels = dbscan.fit_predict(X_outliers.cpu().numpy())
            labels_tensor = torch.tensor(labels, device=self.device)
            
            # Filter out the pure noise (DBSCAN labels noise as -1)
            valid_mask = labels_tensor >= 0
            
            if not valid_mask.any():
                print("Shelf full, but only contained scattered noise. Burning shelf.")
            else:
                # Find all unique clusters and their sizes
                valid_labels = labels_tensor[valid_mask]
                unique_clusters, cluster_counts = torch.unique(valid_labels, return_counts=True)
                
                # We lower the threshold so multiple materials can be found simultaneously.
                # Must have at least enough pixels to calculate covariance reliably (e.g., 2x channels)
                # or a small percentage of the shelf.
                MIN_PURE_PIXELS = max(int(self.outlier_update_treshold * 0.15), 2 * self.n_channels)
                
                components_birthed = 0
                
                # LOOP OVER ALL CLUSTERS FOUND BY DBSCAN
                for cluster_idx, size in zip(unique_clusters, cluster_counts):
                    if size.item() >= MIN_PURE_PIXELS:
                        pure_material_mask = (labels_tensor == cluster_idx)
                        X_pure = X_outliers[pure_material_mask]

                        print(f"\n--- Spawning Component for Cluster {cluster_idx.item()} ---")
                        print(f"DBSCAN isolated {size.item()} pure pixels.") 

                        global_q = self.MFA.q

                        # ---------------------------------------------------------
                        # THE BAYESIAN SPAWNER
                        # ---------------------------------------------------------
                        with torch.no_grad():
                            bayesian_spawner = BayesianMFA_Initializer(
                                n_components=1, 
                                n_channels=self.MFA.D, 
                                q_max=global_q, 
                                max_iter=30, 
                                device=self.device
                            )
                            
                            bayesian_spawner.fit_with_ard(X_pure)
                            
                            #surviving_factors = (bayesian_spawner.alpha[0] < 1e4).sum().item()
                            #print(f"ARD Spawner activated {surviving_factors} effective factors out of {global_q} available.")

                            N_pure = X_pure.shape[0]
                            new_S0 = torch.tensor([1.0], dtype=torch.float32, device=self.device)
                            new_S1 = X_pure.mean(dim=0, keepdim=True)            
                            new_S2 = (X_pure.T @ X_pure).unsqueeze(0) / N_pure  

                            self.MFA.add_component(
                                new_mu=bayesian_spawner.mu.data,
                                new_Lambda=bayesian_spawner.Lambda.data, 
                                new_log_psi=bayesian_spawner.log_psi.data,
                                new_S0=new_S0,
                                new_S1=new_S1,
                                new_S2=new_S2
                            )
                            components_birthed += 1
                        # ---------------------------------------------------------
                
                if components_birthed > 0:
                    print(f"\nSuccessfully birthed {components_birthed} new components this cycle.")
                    # 1. Evaluate the data on the updated MFA only ONCE after all new components are added
                    with torch.no_grad():
                        log_resp_norm_new_data, _, log_probs_new, _ = self.MFA.e_step(X)
                else:
                    print(f"Clusters found, but none met the minimum size threshold ({MIN_PURE_PIXELS} pixels). Burning shelf.")

            # Burn the shelf (happens regardless of whether components were birthed)
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

