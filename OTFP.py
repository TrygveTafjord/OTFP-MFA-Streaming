import torch
import numpy as np
import glob
from MFA import MFA


class MFA_OTFP:
    def __init__(self, n_channels: int, outlier_significance: float, device: str, outlier_update_treshold: int, L2_normalization: bool = True, q_max: int = 5):
        # System Parameters
        self.device = device
        self.outlier_update_treshold = outlier_update_treshold
        self.L2_normalization = L2_normalization

        # Train Initial Model and perform model selection        
        data_dir = glob.glob(f'data/training_l1b/*.nc')
        initil_data = fetch_init_data(dir = data_dir, num_samples = 20000) #Not yet implemented, just a placeholder for fetching initial data for model selection and initial fitting, will probably be implemented as a method in this class later on, or perhaps even as a separate utility-function or perhaps the data will be an input to the constructor, we will see.
        K, q = self.perform_model_selection(data=initil_data, n_channels=n_channels, q_max=q_max)
        
        self.MFA = MFA(n_components = K, n_channels = n_channels, n_factors=q).to(device)        
        self.MFA.fit(initil_data) # Fit the initial model on the initial data
        
        del initil_data

        # Outliers are saved to the shelf
        self.outliers_shelf = torch.empty((self.outlier_update_treshold, n_channels), device=self.device, dtype=torch.float32)
        # System State
        self.n_samples_seen = 0
        self.num_outliers_on_shelf = 0
        self.current_idx_in_chunk = 0
        self.batch_file_counter = 0
        self.n_model_updates = 0

    
    def perform_model_selection(self, data, n_channels, q_max):
        # Dummy implementation
        K = 8      
        q = min(q_max, n_channels)  
        return K, q

    
    def process_data_block(self, X):

        log_resp_norm, log_likelihood = self.MFA.e_step(X)

        
        # Update counter
        self.n_samples_seen += X.shape[0]



    def save_current_chunk(self):
        """Saves the valid data in the current buffers to a file."""
        if self.current_idx_in_chunk == 0:
            return # Nothing to save

        # Get views of the valid data 
        valid_T = self.T_buffer[:self.current_idx_in_chunk]
        valid_res = self.res_buffer[:self.current_idx_in_chunk]
        valid_T2 = self.T2_buffer[:self.current_idx_in_chunk]
        valid_F_res = self.F_res_buffer[:self.current_idx_in_chunk]
        valid_Q_res = self.Q_buffer[:self.current_idx_in_chunk]
        
        dir = "./data/processed/"
        filename = f"CIM_batch_full_{self.batch_file_counter}.pt"
        #print(f"Saving {self.current_idx_in_chunk} samples to {filename}")
        save_loaction = dir + filename

        # This a blocking I/O operation.
        torch.save({
            'T': valid_T.cpu(), # Move to CPU for saving
            'Residuals': valid_res.cpu(),
            'T2': valid_T2.cpu(),
            'F_res': valid_F_res.cpu(),
            'Q_res': valid_Q_res.cpu(), 
        }, save_loaction)









