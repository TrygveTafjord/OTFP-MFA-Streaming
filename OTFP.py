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
        self.ll_threshold = 0.0  # Calibrated after initial fit
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

        print(f"Finished setup of OTFP-MFA: K={self.MFA.K}, q={self.MFA.q}, Log-likelyhood threshold={self.ll_threshold} ")


    def _run_mfa_setup(self, X):
        
        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        with torch.no_grad():
            self.MFA.initialize_parameters(X)
            self.MFA.fit(X) # Fit the initial model on the initial data
            _, init_ll, _ = self.MFA.e_step(X)
        
        kth_value = max(1, int(self.outlier_significance * X.shape[0]))
        self.ll_threshold = torch.kthvalue(init_ll, kth_value).values.item()
        return
    

    def _perform_model_selection(self, data, n_channels, q_max):
        # Dummy implementation
        K = 8      
        q = 5
        return K, q

    def process_data_block(self, X):

        # 0. The 100x Guardrail for Pylance and Runtime Safety
        if self.MFA is None:
            raise RuntimeError("CRITICAL: MFA model was not initialized. Ensure _run_setup_routine completed successfully.")

        if self.L2_normalization:
            X = torch.nn.functional.normalize(X, p=2, dim=1)

        self.n_samples_seen += X.shape[0]
        
        with torch.no_grad():
            # Unpack the new geometric fits (log_probs) alongside the Bayesian repsonsibilities
            log_resp_norm, log_likelihood, log_probs = self.MFA.e_step(X)

        # -----------------------------------------------------------------
        # 1. OUTLIER DETECTION & NEW MATERIAL DISCOVERY (The Global Guardrail)
        # -----------------------------------------------------------------
        outlier_mask = log_likelihood < self.ll_threshold
        num_new_outliers = outlier_mask.sum().item()
        
        if num_new_outliers + self.num_outliers_on_shelf > self.outlier_update_treshold:
            # --- SHELF IS FULL: TRIGGER NEW COMPONENT ADDITION ---
            X_outliers = self.global_outliers_shelf[:self.num_outliers_on_shelf]
            X_outliers = torch.cat([X_outliers, X[outlier_mask]], dim=0)

            # Perform INSTANT model selection for q using cumulative variance
            cov_matrix = torch.cov(X_outliers.T)
            eigenvalues = torch.linalg.eigvalsh(cov_matrix)
            eigenvalues = torch.flip(eigenvalues, dims=[0])

            cumulative_variance = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
            q_new = torch.searchsorted(cumulative_variance, 0.95).item() + 1
            q_new = min(q_new, self.q_max)

            print(f"Adding 1 new component with {q_new} latent factors.")

            # Fit the localized MFA on the outlier repository
            temp_mfa = MFA(n_components=1, n_channels=self.MFA.D, n_factors=q_new, device=self.device)
            temp_mfa.initialize_parameters(X_outliers)
            temp_mfa.fit(X_outliers)

            # Calculate the initial weight of the new component
            new_weight = X_outliers.shape[0] / max(1, self.n_samples_seen)

            # Merge into the global model
            self.MFA.add_component(
                new_mu=temp_mfa.mu.data,
                new_Lambda=temp_mfa.Lambda.data,
                new_log_psi=temp_mfa.log_psi.data,
                new_weight=new_weight
            )

            # Expand tracking tensors for the newly created component
            self.component_pixel_counts = torch.cat([self.component_pixel_counts, torch.tensor([0], device=self.device)])
            self.local_shelf_counts = torch.cat([self.local_shelf_counts, torch.tensor([0], device=self.device)])
            self.local_shelves[self.MFA.K - 1] = []

            # Evaluate new threshold (Note: we use the temp_mfa to get its pure threshold)
            with torch.no_grad():
                _, new_ll, _ = temp_mfa.e_step(X_outliers)
                
            kth_value = max(1, int(self.outlier_significance * X_outliers.shape[0]))
            new_component_threshold = torch.kthvalue(new_ll, kth_value).values.item()
            
            # NOTE: For your thesis, consider tracking local thresholds rather than lowering the global one!
            # For now, keeping your implementation:
            old_threshold = self.ll_threshold
            self.ll_threshold = min(self.ll_threshold, new_component_threshold)
            
            print(f"Updated LL Threshold from {old_threshold:.2f} to {self.ll_threshold:.2f} to accommodate new material.")

            self.num_outliers_on_shelf = 0
            self.n_model_updates += 1

        elif num_new_outliers > 0:
            # --- SHELF NOT FULL: JUST ADD OUTLIERS ---
            start_idx = self.num_outliers_on_shelf
            end_idx = start_idx + num_new_outliers
            self.global_outliers_shelf[start_idx : end_idx] = X[outlier_mask]
            self.num_outliers_on_shelf += num_new_outliers
        
        # -----------------------------------------------------------------
        # 2. INBOUND PIXEL PROCESSING (Assignment & Drift Detection)
        # -----------------------------------------------------------------
        inbound_mask = ~outlier_mask
        if not inbound_mask.any():
            return # Entire block was outliers, nothing more to do

        X_in = X[inbound_mask]
        log_resp_norm_in = log_resp_norm[inbound_mask]
        log_probs_in = log_probs[inbound_mask] # NEW: Extract the pure geometric fits
        resp_in = torch.exp(log_resp_norm_in) 
        
        # A. The Purity Guardrail (Shannon Entropy)
        # We still use the full posterior (resp_in) to check if the model is confused
        entropy = -torch.sum(resp_in * log_resp_norm_in, dim=1)
        
        purity_threshold = 0.5
        pure_mask = entropy < purity_threshold
        
        X_pure = X_in[pure_mask]
        
        # B. Geometric Component Assignment (Bypassing the Prior Trap)
        # We assign components based purely on physical shape (log_probs_in)
        best_components = torch.argmax(log_probs_in[pure_mask], dim=1)
        
        # Extract the pure unnormalized log-probability directly! No reverse engineering needed.
        # This acts as our Mahalanobis / Hotelling's T^2 proxy
        log_prob_pure = log_probs_in[pure_mask, best_components]
        
        # C. Route pure pixels to Local Shelves or skip them
        for k in range(self.MFA.K):
            k_mask = (best_components == k)
            if not k_mask.any():
                continue
                
            X_k = X_pure[k_mask]
            log_prob_k = log_prob_pure[k_mask]
            
            # --- DRIFT DETECTION ---
            # TODO: Replace this hardcoded +50 with a dynamic component-specific tracking system
            local_drift_threshold = self.ll_threshold + 50
            
            drifting_mask = log_prob_k < local_drift_threshold
            
            if drifting_mask.any():
                X_drifting = X_k[drifting_mask]
                self.local_shelves[k].append(X_drifting)
                self.local_shelf_counts[k] += X_drifting.shape[0]
                
                # --- LOCAL SHELF IS FULL: UPDATE COMPONENT ---
                if self.local_shelf_counts[k] >= self.outlier_update_treshold:
                    X_update = torch.cat(self.local_shelves[k], dim=0)
                    print(f"Component {k} drifting! Updating factors using {X_update.shape[0]} pixels...")
                    
                    self.MFA.update_single_component(k, X_update)
                    
                    self.local_shelves[k] = []
                    self.local_shelf_counts[k] = 0
                
            # --- PERFECTLY EXPLAINED PIXELS ---
            perfect_mask = ~drifting_mask
            if perfect_mask.any():
                self.component_pixel_counts[k] += perfect_mask.sum().item()

        # -----------------------------------------------------------------
        # 3. GLOBAL STATE UPDATES
        # -----------------------------------------------------------------
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









