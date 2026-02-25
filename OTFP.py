import torch
from mfa import MFA

class MFA_OTFP:
    def __init__(self, init_data: torch.Tensor, n_channels: int, outlier_significance: float, device: str, outlier_update_treshold: int, L2_normalization: bool = True, q_max: int = 5):
        # System Parameters
        self.device = device
        self.outlier_update_treshold = outlier_update_treshold
        self.outlier_significance = outlier_significance
        self.L2_normalization = L2_normalization
        self.q_max = q_max

        if self.L2_normalization:
            init_data = torch.nn.functional.normalize(init_data, p=2, dim=1)
        
        #start model selection, this will be implemented as a method in this class, but for now it is just a dummy function that returns fixed values for K and q, we will implement the actual model selection logic later on, perhaps using BIC or AIC or some other criterion to select the best K and q based on the initial data.
        K, q = self.perform_model_selection(data=init_data, n_channels=n_channels, q_max=q_max)

        #Fit the model
        print("Creating initial model")
        self.MFA = MFA(n_components = K, n_channels = n_channels, n_factors=q).to(device)        
        with torch.no_grad():
            self.MFA.fit(init_data) # Fit the initial model on the initial data
        print("initial model created")

        with torch.no_grad():
            _, init_ll = self.MFA.e_step(init_data)
        
        kth_value = max(1, int(outlier_significance * init_data.shape[0]))
        self.ll_threshold = torch.kthvalue(init_ll, kth_value).values.item()
        print(f"Calculated Global Outlier Log-Likelihood Threshold: {self.ll_threshold:.2f}")

        # Outliers are saved to the shelf, allocate memory
        self.outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32)
        self.pixels_assigned_per_component = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32)

        # Dictionary to hold drifting pixels for each specific component
        self.local_shelves = {k: [] for k in range(self.MFA.K)}
        self.local_shelf_counts = torch.zeros(self.MFA.K, dtype=torch.long, device=self.device)
        
        # Track perfectly explained pixels (Used to update component weights later)
        self.component_pixel_counts = torch.zeros(self.MFA.K, dtype=torch.long, device=self.device)
        
        # System State
        self.n_samples_seen = 0
        self.num_outliers_on_shelf = 0
        self.n_model_updates = 0
        self.blocks_processed = 0
    
    def perform_model_selection(self, data, n_channels, q_max):
        # Dummy implementation
        K = 8      
        q = min(q_max, n_channels)  
        return K, q

    def process_data_block(self, X):

        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        self.n_samples_seen += X.shape[0]
        with torch.no_grad():
            log_resp_norm, log_likelihood = self.MFA.e_step(X)

        responsibilities = torch.exp(log_resp_norm)
        new_cluster_ids = torch.argmax(responsibilities, dim=1)

        outlier_mask = log_likelihood < self.ll_threshold
        
        num_new_outliers = outlier_mask.sum().item()
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            # If shelf is full, update the model with outliers
            # update the model with the outliers on the shelf, this logic will be implemented as a method, but the exact detaljes are not yet determined
            X_outliers = self.outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0) # Add the new outliers to the outliers on the shelf

            # Update routine, I will: 
            #1 Perform model selection on the outlier-repo
            #2 Fit an MFA on the outlier repo
            #3 Merge the MFA into the existing model by adding the components
            #4 Calculate distance to the other components, to see if the new component is really new, or if it is just a small update to an existing component, 
            # if it is just a small update, then I will merge it with the existing component instead of adding a new one, 
            # this will be done by calculating the distance between the new component and the existing components, and if the distance is below a certain threshold, 
            # then I will merge it with the existing component, otherwise I will add it as a new component.

            # 2. Perform INSTANT model selection for q
            cov_matrix = torch.cov(X_outliers.T)
            eigenvalues = torch.linalg.eigvalsh(cov_matrix)
            eigenvalues = torch.flip(eigenvalues, dims=[0]) # Sort descending

            cumulative_variance = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
            # Find the first index where cumulative variance > 0.99
            q_new = torch.searchsorted(cumulative_variance, 0.99).item() + 1

            # Bound q to your maximum
            q_new = min(q_new, self.q_max)

            print(f"Adding 1 new component with {q_new} latent factors.")

            temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
            # Use your K-means initialization to give it a good start
            temp_mfa.initialize_parameters(X_outliers)
            temp_mfa.fit(X_outliers)

            # 4. Calculate the weight of the new component
            # Physically: "What percentage of the TOTAL data we have seen so far is this new material?"
            new_weight = X_outliers.shape[0] / max(1, self.n_samples_seen)

            # 5. Merge into the global model
            self.MFA.add_component(
                new_mu=temp_mfa.mu.data,
                new_Lambda=temp_mfa.Lambda.data,
                new_log_psi=temp_mfa.log_psi.data,
                new_weight=new_weight
            )

            # Expand tracking tensors for the newly created component
            self.component_pixel_counts = torch.cat([self.component_pixel_counts, torch.tensor([0], device=self.device)])
            self.local_shelf_counts = torch.cat([self.local_shelf_counts, torch.tensor([0], device=self.device)])
            
            # Add a new shelf for the component
            self.local_shelves[self.MFA.K - 1] = []

            with torch.no_grad():
                _, new_ll = self.MFA.e_step(X_outliers)
                
            kth_value = max(1, int(self.outlier_significance * X_outliers.shape[0]))
            new_component_threshold = torch.kthvalue(new_ll, kth_value).values.item()
            
            # Expand the global tolerance if the new component is naturally "wider" (lower density)
            old_threshold = self.ll_threshold
            self.ll_threshold = min(self.ll_threshold, new_component_threshold)
            
            print(f"Updated LL Threshold from {old_threshold:.2f} to {self.ll_threshold:.2f} to accommodate new material.")

            self.num_outliers_on_shelf = 0
            self.n_model_updates += 1

        elif num_new_outliers > 0:
            
            start_idx = self.num_outliers_on_shelf
            end_idx = start_idx + num_new_outliers
            
            self.outliers_shelf[start_idx : end_idx] = X[outlier_mask]
            
            self.num_outliers_on_shelf += num_new_outliers
        
        # 1. Isolate pixels that are not global outliers
        inbound_mask = ~outlier_mask
        if not inbound_mask.any():
            return # Entire block was outliers, nothing more to do

        X_in = X[inbound_mask]
        resp_in = responsibilities[inbound_mask]
        log_resp_norm_in = log_resp_norm[inbound_mask]
        
        # 2. Isolate "clean" pixels (The Purity Guardrail)
        # Shannon entropy: H = -sum(p * log(p)). Low entropy = high purity.
        entropy = -torch.sum(resp_in * log_resp_norm_in, dim=1)
        
        # Pixels with entropy near 0 are overwhelmingly assigned to a single component
        purity_threshold = 0.5 # You can tune this. Lower = stricter purity
        pure_mask = entropy < purity_threshold
        
        # Extract the pure pixels and find their assigned component
        X_pure = X_in[pure_mask]
        best_components = torch.argmax(resp_in[pure_mask], dim=1)
        
        # Extract the actual unnormalized log-probability (Proxy for Mahalanobis / Hotelling's T^2)
        # log_resp_norm = log_prob + log_pi - log_likelihood -> log_prob = log_resp_norm + log_likelihood - log_pi
        log_prob_pure = log_resp_norm_in[pure_mask, best_components] + log_likelihood[inbound_mask][pure_mask] - self.MFA.log_pi[best_components]

        # 3. Route pure pixels to Local Shelves or skip them
        # We need a dynamic update threshold. For now, we assume if the log_prob is below the 
        # top 50% of the component's normal fit, it's "drifting" and needs an update.
        
        for k in range(self.MFA.K):
            # Find pure pixels belonging to component K
            k_mask = (best_components == k)
            if not k_mask.any():
                continue
                
            X_k = X_pure[k_mask]
            log_prob_k = log_prob_pure[k_mask]
            
            # --- DRIFT DETECTION ---
            # If we don't have a local threshold yet, use a placeholder (you will want to track this dynamically per component later)
            local_drift_threshold = self.ll_threshold + 50 # Example: slightly higher than global threshold
            
            drifting_mask = log_prob_k < local_drift_threshold
            if drifting_mask.any():
                X_drifting = X_k[drifting_mask]
                self.local_shelves[k].append(X_drifting)
                self.local_shelf_counts[k] += X_drifting.shape[0]
                
                # 5. If the repo is full -> update the component!
                if self.local_shelf_counts[k] >= self.outlier_update_treshold:
                    # Stitch the shelf together
                    X_update = torch.cat(self.local_shelves[k], dim=0)
                    
                    print(f"Component {k} drifting! Updating factors using {X_update.shape[0]} pixels...")
                    
                    # Perform the Online EM update for this specific component
                    self.MFA.update_single_component(k, X_update)
                    
                    # Empty the shelf
                    self.local_shelves[k] = []
                    self.local_shelf_counts[k] = 0
                
            # 6. Perfectly Explained Pixels -> Do not update the FA! 
            perfect_mask = ~drifting_mask
            if perfect_mask.any():
                # Just keep track of the count for future weight recalculations
                self.component_pixel_counts[k] += perfect_mask.sum().item()

        # Check what clusters each well-explained pixel belongs to
        
        #1 Isolate pixels that are well explained by the model (high log_likelyhood)
        
        #2 Isolate "clean" pixels that are only well explained by one of the components, this is done by calculating the Shannon entropy of your responsibilities matrix for each pixel

        #4 If the pixel is only somewhat well explained, add it to a component update - repository <- I might drop updates! Or I could use hotellings t^2 to find pixels that hace the correct covarianse-structure.
        
        #5 If the repo is full -> update the component (perhaps by performing weighted bayesian inference?)

        #6 If the pixel is really well explained, increase the count of well explained pixels
        self.blocks_processed += 1
        
        # Update the mixing weights every 100 blocks (tune this based on your data rate)
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









