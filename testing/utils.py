import torch
import torch.nn.functional as F
from hypso import Hypso
import numpy as np


def calculate_rmse(original, reconstructed):
    """Calculates Root Mean Squared Error."""
    mse = torch.mean((original - reconstructed) ** 2, dim=1)
    return torch.sqrt(mse).mean().item()

def calculate_sam(original, reconstructed, epsilon=1e-8):
    """
    Calculates Spectral Angle Mapper (SAM) in radians.
    Formula: arccos( (x . y) / (|x| * |y|) )
    """
    # Normalize vectors to unit length
    norm_orig = torch.norm(original, p=2, dim=1, keepdim=True)
    norm_recon = torch.norm(reconstructed, p=2, dim=1, keepdim=True)
    
    # Dot product
    dot_product = torch.sum(original * reconstructed, dim=1, keepdim=True)
    
    # Cosine similarity
    cosine_sim = dot_product / (norm_orig * norm_recon + epsilon)
    
    # Clamp to avoid numerical issues slightly outside [-1, 1]
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
    
    sam_rad = torch.acos(cosine_sim)
    return sam_rad.mean().item()


def get_data(data_dir, data_product, target_total_samples):
    
    len_data_dir = len(data_dir)
    samples_per_file = target_total_samples // len_data_dir  
    sampled_data_list = []

    print(f"Aiming to extract ~{samples_per_file} pixels per file from {len_data_dir} files to reach a total of ~{target_total_samples} samples.")

    i = 1
    for file in data_dir:
        # Load Data
        try:
            satobj = Hypso(file) 
            if satobj is None: continue

            # Load and reshape
            match data_product:
                case 'l1a':
                    data = satobj.l1a_cube.values.astype(np.float32)
                case 'l1b':
                    data = satobj.l1b_cube.values.astype(np.float32)
                case 'l1d':
                    data = satobj.l1d_cube.values.astype(np.float32)
                case _:
                    raise ValueError(f"Unknown data product: {data_product}")

            h, w, b = data.shape
            data_2d = data.reshape(-1, b) # Shape: (Total_Pixels_In_Image, 120)

            # Random Subsampling 
            total_pixels_in_image = data_2d.shape[0]

            # Determine how many to take (don't take more than exists)
            n_to_take = min(samples_per_file, total_pixels_in_image)

            # Generate random indices
            rng = np.random.default_rng()
            indices = rng.choice(total_pixels_in_image, size=n_to_take, replace=False)

            # Grab the random pixels and add to list
            sampled_pixel_subset = data_2d[indices, :]
            sampled_data_list.append(sampled_pixel_subset)

            print(f"{i}/{len_data_dir} | File: {file} | Extracted {n_to_take} pixels.")
            i += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
        
    data = np.concatenate(sampled_data_list, axis=0)
    #Convert to torch tensor
    print("-" * 30)
    print(f"Final Analysis Dataset Shape: {data.shape}")

    return data 

def reconstruct_mfa(model, X):
    """Reconstructs X using the Component-Specific posterior expectation."""
    with torch.no_grad():
        log_resp, _ = model.e_step(X) 
        responsibilities = torch.exp(log_resp) 
        cluster_ids = torch.argmax(responsibilities, dim=1)
        X_rec = torch.zeros_like(X)
        
        for k in range(model.K):
            mask = (cluster_ids == k)
            if mask.sum() == 0: continue
            
            X_k = X[mask]
            X_k_centered = X_k - model.mu[k]
            Lambda = model.Lambda[k] 
            
            psi_k = torch.exp(model.log_psi[k])
            Psi_inv = torch.diag(1.0 / psi_k) 
            
            M = torch.inverse(torch.eye(model.q, device=X.device) + Lambda.T @ Psi_inv @ Lambda)
            beta = M @ Lambda.T @ Psi_inv
            z_k = (X_k_centered @ beta.T)
            
            X_rec[mask] = z_k @ Lambda.T + model.mu[k]
            
    return X_rec, cluster_ids



def mean_center_data(data):
    """Mean centers the data."""
    mean = torch.mean(data, dim=0, keepdim=True)
    centered_data = data - mean
    
    return centered_data, mean

def l2_normalize_data(data, epsilon=1e-8):
    """Performs L2 normalization on the data."""
    norms = torch.norm(data, p=2, dim=1, keepdim=True)
    normalized_data = data / (norms + epsilon)
    return normalized_data