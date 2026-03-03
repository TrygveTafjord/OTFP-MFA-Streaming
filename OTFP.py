import torch
from mfa import MFA

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
        self.MFA = MFA(n_components = K, n_channels = n_channels, n_factors=q).to(self.device)
        self.local_thresholds = torch.zeros(self.MFA.K, device=self.device, dtype=torch.float32) #We detect outliers based on component-tresholds        
        self._run_mfa_setup(init_data)
        # Streaming statistics
        self.n_samples_seen = 0
        self.n_model_updates = 0
        self.blocks_processed = 0

        # Memory buffers
        # Holds outliers that are not well explained by the model
        self.global_outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32) 
        self.num_outliers_on_shelf = 0
        
        # Local shelves for drifting known materials (Components)
        self.local_shelf_counts = torch.zeros(self.MFA.K, dtype=torch.long, device=self.device) # Tensor tracking size per component     
        self.component_pixel_counts = torch.zeros(self.MFA.K, dtype=torch.long, device=self.device) # Counter for perfectly explained pixels (for weight re-calibration)
        
        # Initialize dictionary for local shelves
        self.local_shelves = {k: [] for k in range(self.MFA.K)} # Dict mapping component_id -> List of tensors

        print(f"Finished setup of OTFP-MFA: K={self.MFA.K}, q={self.MFA.q}")


    def _run_mfa_setup(self, X):
        
        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        with torch.no_grad():
            self.MFA.initialize_parameters(X)
            self.MFA.fit(X) 
            
            # log_probs is the raw, unweighted geometric fit. No priors. No rarity.
            _, _, log_probs = self.MFA.e_step(X)
        
        # Assign based purely on which material's shape the pixel fits best
        best_components = torch.argmax(log_probs, dim=1)
        
        # Forge the boundaries based strictly on percentiles
        for k in range(self.MFA.K):
            k_mask = (best_components == k)
            k_log_probs = log_probs[k_mask, k]
            
            num_pixels_assigned = k_log_probs.shape[0]
            
            if num_pixels_assigned > 10: 
                # INITIALIZATION GUARDRAIL: We use a stricter baseline (e.g., 5%) for the first batch. 
                strict_significance = self.outlier_significance 
                kth_value = max(1, int(strict_significance * num_pixels_assigned))
                
                # The kth_value is our hard boundary. 
                self.local_thresholds[k] = torch.kthvalue(k_log_probs, kth_value).values.item()
                print(f"Component {k} initialized. Pixels: {num_pixels_assigned}. Threshold: {self.local_thresholds[k]:.2f}")
            else:
                # If a component captures almost nothing, it's a ghost. We give it a low threshold
                # so it doesn't accidentally swallow real anomalies later.
                self.local_thresholds[k] = -1e5
                print(f"Warning: Component {k} is a ghost (only {num_pixels_assigned} pixels).")
                
        return

    def _perform_model_selection(self, data, n_channels, q_max):
        # Dummy implementation
        K = 6      
        q = 5
        return K, q

    def process_data_block(self, X):
         # SWITCH TO USING log_likelyhood for detecting outliers!!! -> Or try at least...

        if self.MFA is None:
            raise RuntimeError("CRITICAL: MFA model was not initialized.")

        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        self.n_samples_seen += X.shape[0]
        
        with torch.no_grad():
            # Extract log_probs (raw geometry). We ignore log_resp_norm and log_likelihood.
            _, _, log_probs = self.MFA.e_step(X)

        max_log_probs, best_components = torch.max(log_probs, dim=1)
        assigned_thresholds = self.local_thresholds[best_components]
        
        outlier_mask = max_log_probs < assigned_thresholds
        num_new_outliers = outlier_mask.sum().item()

        inlier_mask = ~outlier_mask
        num_inliers = inlier_mask.sum().item()

        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            # Isolate the signal from the noise
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(X_outliers.cpu().numpy())
            
            labels_tensor = torch.tensor(labels, device=self.device)
            cluster_counts = torch.bincount(labels_tensor)
            dominant_cluster_idx = torch.argmax(cluster_counts).item()
            dominant_size = cluster_counts[dominant_cluster_idx].item()
            
            # Don't model the static. If the dominant cluster is less than 15% of the shelf, it's a ghost.
            MIN_PURE_PIXELS = int(self.outlier_update_treshold * 0.15)
            
            if dominant_size >= MIN_PURE_PIXELS:
                pure_material_mask = (labels_tensor == dominant_cluster_idx)
                X_pure = X_outliers[pure_material_mask]

                print(f"Shelf full. Isolated {X_pure.shape[0]} pure pixels. Birthing new component.") 

                cov_matrix = torch.cov(X_pure.T)
                eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                eigenvalues = torch.flip(eigenvalues, dims=[0])

                cumulative_variance = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
                q_new = torch.searchsorted(cumulative_variance, 0.95).item() + 1
                print(f"Needed: {q_new} PCs to describe 95% of variance in the setup")
                
                if q_new <= self.q_max:

                    temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
                    temp_mfa.initialize_parameters(X_pure)
                    temp_mfa.fit(X_pure)

                    new_weight = X_pure.shape[0] / max(1, self.n_samples_seen)

                    self.MFA.add_component(
                        new_mu=temp_mfa.mu.data,
                        new_Lambda=temp_mfa.Lambda.data,
                        new_log_psi=temp_mfa.log_psi.data,
                        new_weight=new_weight
                    )

                    self.component_pixel_counts = torch.cat([self.component_pixel_counts, torch.tensor([0], device=self.device)])
                    self.local_shelf_counts = torch.cat([self.local_shelf_counts, torch.tensor([0], device=self.device)])
                    self.local_shelves[self.MFA.K - 1] = []

                    # Calculate the boundary for this specific new material
                    with torch.no_grad():
                        _, _, new_log_probs = temp_mfa.e_step(X_pure)

                    kth_value = max(1, int(self.outlier_significance * X_pure.shape[0]))
                    new_component_threshold = torch.kthvalue(new_log_probs[:, 0], kth_value).values.item()

                    self.local_thresholds = torch.cat([
                        self.local_thresholds, 
                        torch.tensor([new_component_threshold], device=self.device, dtype=torch.float32)
                    ])

                    print(f"New material boundary set in stone at LL: {new_component_threshold:.2f}")
                    self.n_model_updates += 1
                else:
                    print("Needed to many components to fit the shelf, it is just noise!")
            else:
                print(f"Shelf full, but dominant cluster only has {dominant_size} pixels. Just ghosts and noise.")

            # --- THE CLEANSING FIRE ---
            # Burn the shelf. We don't put the leftovers back. We wait for fresh evidence.
            self.num_outliers_on_shelf = 0

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
    
    def update_global_weights(self):
        """
        Periodically recalibrates the global mixing weights (pi) based on the 
        actual distribution of materials seen by the satellite.
        """
        # 1. Combine all known pixels assigned to each component
        # This includes the "perfect" pixels and the drifting pixels currently on the shelves
        total_counts = self.component_pixel_counts + self.local_shelf_counts
        
        # 2. Apply Laplace Smoothing to prevent log(0) for new/rare components
        smoothed_counts = total_counts + 1.0 
        
        # 3. Calculate the new linear probabilities
        new_pi = smoothed_counts / smoothed_counts.sum()
        
        # 4. Safely inject the new weights into the MFA model in log-space
        with torch.no_grad():
            self.MFA.log_pi.data = torch.log(new_pi)









