import torch
import glob
from mfa import MFA
from utils import get_data
from enum import Enum
import os
import time


class DataProduct(Enum):
    L1A = 'l1a'
    L1B = 'l1b'
    L1D = 'l1d'


data_product = DataProduct.L1B
K = 9  # Number of MFA compobnents to use 
q = 10 # Latent dimensionality for MFA
Train_PCA_on_L2Normalized = False
Train_MFA_on_L2Normalized = True


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = glob.glob(f'../data/training_{data_product.value}/*.nc')
print(f"Found {len(data_dir)} files.")


target_total_samples = 200000

data_list = get_data(data_dir, data_product.value, target_total_samples)

data = torch.from_numpy(data_list).float().to(device)


if Train_MFA_on_L2Normalized:
    print("Applying L2 Normalization to data")
    norms = torch.norm(data, p=2, dim=1, keepdim=True)
    epsilon = 1e-8
    X = data / (norms + epsilon)
else: 
    print("Using raw data without normalization")
    X = data
# Ensure float32
X = X.float()
print(f"Shape of X: {X.shape}")

mfa_model = MFA(n_components=K, n_channels=X.shape[1], n_factors=q)

print("Initializing MFA")
mfa_model.initialize_parameters(X)
print("Training MFA")

start_time = time.perf_counter()
mfa_model.fit(X)
end_time = time.perf_counter()
training_time = start_time - end_time
print(f"Training time is: {training_time}s")


# Clean up memory if using GPU
if device.type == 'cuda':
    torch.cuda.empty_cache()

# 2. Build the state dictionary
mfa_state = {
    'model_state_dict': mfa_model.state_dict(), 
    
    # Hyperparameters required to initialize the MFA class
    'hyperparameters': {
        'n_components': mfa_model.K,
        'n_features': mfa_model.D,
        'n_factors': mfa_model.q
    },
    'traininginfo': {
        'n_training_pixels' : target_total_samples,
        'training_time' : training_time 
    }
    }

torch.save(mfa_state, f'models/mfa.pt')
    


print(f"Training PCA model")

# Mean Centering
if Train_PCA_on_L2Normalized:
    print("Training PCA on L2-Normalized data")
    X_train = X # X is the L2-Normalized data from above
else: 
    print("Training PCA on original data (not L2-Normalized)")
    X_train = data.float() # Use original data for PCA

mean_vector = X_train.mean(dim=0)
X_centered = X_train - mean_vector

# Compute Covariance Matrix
n_samples = X_centered.shape[0]
cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

# Eigendecomposition (PCA)
eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

# Sort indices in descending order (eigh returns ascending)
sorted_indices = torch.argsort(eigenvalues, descending=True)
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_components = eigenvectors[:, sorted_indices]

# Calculate Cumulative Variance
total_variance = sorted_eigenvalues.sum()
explained_variance_ratio = sorted_eigenvalues / total_variance
cum_var = torch.cumsum(explained_variance_ratio, dim=0)

# Find the threshold (99.5%)
threshold = 0.995
n_components_995 = (cum_var >= threshold).nonzero(as_tuple=False)[0].item() + 1

print("-" * 30)
print(f"Cumulative Variance for top 10 components:\n {cum_var[:10]}")
print("-" * 30)
print(f"Number of components to explain {threshold*100}% variance: {n_components_995}")
print(f"Ceiling for q (latent factors) set to: {n_components_995}")

# 7. Save the PCA Model
pca_state = {
    'components': sorted_components[:, :n_components_995], # Save only needed components
    'mean': mean_vector,
    'explained_variance': sorted_eigenvalues[:n_components_995],
    'n_components_995': n_components_995
}

torch.save(pca_state, f'models/pca.pt')
print(f"PCA model saved to 'models/pca.pt'")
