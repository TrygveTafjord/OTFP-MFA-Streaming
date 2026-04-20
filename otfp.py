from scipy.stats import chi2
import torch
from mfa import MFA
from bayesian_model_selector import BayesianMFA_Initializer
from scipy.ndimage import label, generate_binary_structure


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

    def process_data_block(self, X_3d):
        # Flatten the spatial dimensions for the MFA math
        H, W, D = X_3d.shape
        X_flat = X_3d.view(-1, D) # Shape: (H*W, D)

        if self.L2_normalization:
            X_flat = torch.nn.functional.normalize(X_flat, p=2, dim=1)

        self.n_samples_seen += (H * W)
        
        with torch.no_grad():
            _, _, _, mahalanobis_dists = self.MFA.e_step(X_flat)
            
        min_mahalanobis, _ = torch.min(mahalanobis_dists, dim=1)
        outlier_mask_1d = min_mahalanobis > self.chi2_threshold
        inlier_mask_1d = ~outlier_mask_1d

        # 2. TRACK INLIERS
        if inlier_mask_1d.any():
            self._process_inliners(X_flat[inlier_mask_1d])

        # 3. SPATIAL OUTLIER BIRTHING
        if outlier_mask_1d.any():
            # Reshape the 1D mask back into the 2D image plane (H, W)
            spatial_mask = outlier_mask_1d.view(H, W).cpu().numpy()
            
            # Find contiguous blobs of outliers (8-way connectivity)
            structure = generate_binary_structure(2, 2)
            labeled_image, num_features = label(spatial_mask, structure=structure) # type: ignore
            
            if num_features > 0:
                self._birth_spatial_components(X_flat, outlier_mask_1d, labeled_image)

    def _process_inliners(self, X_inliers):
        with torch.no_grad():
            log_resp_norm, _, _, _ = self.MFA.e_step(X_inliers)
            self.MFA.stepwise_em_update(X_inliers, log_resp_norm)

    def _birth_spatial_components(self, X_flat, outlier_mask_1d, labeled_image):
        # Convert the (H, W) label array back to a flattened 1D tensor to index X_flat
        labeled_1d = torch.tensor(labeled_image.flatten(), device=self.device)
        
        # Only look at the labels for actual outliers
        valid_labels = labeled_1d[outlier_mask_1d]
        unique_clusters, cluster_counts = torch.unique(valid_labels, return_counts=True)
        
        # Require a physical size footprint to create a new material
        MIN_PURE_PIXELS = 2 * self.n_channels 
        components_birthed = 0
        
        for cluster_idx, size in zip(unique_clusters, cluster_counts):
            if cluster_idx == 0:
                continue # Label 0 is the background (inliers)
                
            if size.item() >= MIN_PURE_PIXELS:
                pure_material_mask = (labeled_1d == cluster_idx)
                X_pure = X_flat[pure_material_mask]
                
                print(f"\n--- Spawning Component for Spatial Object {cluster_idx.item()} ---")
                print(f"Connected Components isolated {size.item()} pure contiguous pixels.") 
                
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
                    
                    # Extract sufficient statistics for the new component
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
        
        if components_birthed > 0:
            print(f"\nSuccessfully birthed {components_birthed} new components from spatial blocks.")
