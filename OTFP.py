import torch
from mfa import MFA

class MFA_OTFP:
    def __init__(self, init_data: torch.Tensor, n_channels: int, outlier_significance: float, device: str, outlier_update_treshold: int, L2_normalization: bool = True, q_max: int = 5):
        # System Parameters
        self.device = device
        self.outlier_update_treshold = outlier_update_treshold
        self.outlier_significance = outlier_significance
        self.L2_normalization = L2_normalization

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
        
        # System State
        self.n_samples_seen = 0
        self.num_outliers_on_shelf = 0
        self.n_model_updates = 0
    
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

            self.num_outliers_on_shelf = 0
            self.n_model_updates += 1

        elif num_new_outliers > 0:
            
            start_idx = self.num_outliers_on_shelf
            end_idx = start_idx + num_new_outliers
            
            self.outliers_shelf[start_idx : end_idx] = X[outlier_mask]
            
            self.num_outliers_on_shelf += num_new_outliers

        # Check what clusters each well-explained pixel belongs to
        
        #1 Isolate pixels that are well explained by the model (high log_likelyhood)
        
        #2 Isolate "clean" pixels that are only well explained by one of the components, this is done by calculating the Shannon entropy of your responsibilities matrix for each pixel

        #4 If the pixel is only somewhat well explained, add it to a component update - repository <- I might drop updates! Or I could use hotellings t^2 to find pixels that hace the correct covarianse-structure.
        
        #5 If the repo is full -> update the component (perhaps by performing weighted bayesian inference?)

        #6 If the pixel is really well explained, increase the count of well explained pixels

        return 










