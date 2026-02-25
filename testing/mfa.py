import torch
import torch.nn as nn
import math

class MFA(nn.Module):
    def __init__(self, n_components, n_channels, n_factors, tol=1e-4, max_iter=150, device='cpu'):
        super().__init__()
        self.K = n_components
        self.D = n_channels
        self.q = n_factors
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        
        # Initialize parameters
        self.log_pi = nn.Parameter(torch.log(torch.ones(self.K, device=self.device) / self.K))
        # Initialize means centered around 0 but spread out
        self.mu = nn.Parameter(torch.randn(self.K, self.D, device=self.device) * 0.1)
        # Factor loadings (K, D, q)
        self.Lambda = nn.Parameter(torch.randn(self.K, self.D, self.q, device=self.device) * 0.1)
        # Log Diagonal noise is specific to each component (K, D)
        self.log_psi = nn.Parameter(torch.log(torch.ones(self.K, self.D, device=self.device) * 1e-2))
        
    def fit(self, X):
        X = X.to(self.device)
        N = X.shape[0]

        prev_ll = -float('inf')
        with torch.no_grad():
            for i in range(self.max_iter):
                # --- E-Step ---
                log_resp, log_likelihood = self.e_step(X)
                current_ll = log_likelihood.mean()
                
                # --- M-Step ---
                resp = torch.exp(log_resp) # (N, K)
                self.m_step(X, resp)
                
                # Convergence check
                diff = current_ll - prev_ll
                if i > 0 and abs(diff) < self.tol:
                    break
                prev_ll = current_ll
                
            self.final_ll = prev_ll * N 
        
    def e_step(self, X):
        log_resps = []
        
        for k in range(self.K):
            L_k = self.Lambda[k] 
            
            # Extract the specific variance for component K
            psi_k = torch.exp(self.log_psi[k]) + 1e-6
            
            # C_k = Lambda @ Lambda.T + Psi_k
            C_k = L_k @ L_k.T + torch.diag(psi_k) 
            
            # Robustness Jitter
            jitter = 1e-5 * torch.eye(self.D, device=self.device)
            C_k = C_k + jitter
            
            try:
                dist = torch.distributions.MultivariateNormal(self.mu[k], covariance_matrix=C_k)
                log_prob = dist.log_prob(X)
            except ValueError:
                # Fallback for numerical instability
                log_prob = torch.ones(X.shape[0], device=self.device) * -1e20
            
            log_resps.append(log_prob + self.log_pi[k])
            
        log_resps = torch.stack(log_resps, dim=1) 
        log_likelihood = torch.logsumexp(log_resps, dim=1) 
        log_resp_norm = log_resps - log_likelihood.unsqueeze(1)
        return log_resp_norm, log_likelihood

    def m_step(self, X, resp):
        N = X.shape[0]
        Nk = resp.sum(dim=0) + 1e-10 
        
        # 1. Update Pi
        self.log_pi.data = torch.log(Nk / N)
        
        # 2. Update Mu
        for k in range(self.K):
            resp_k = resp[:, k].unsqueeze(1) # (N, 1)
            mu_k = (resp_k * X).sum(dim=0) / Nk[k]
            self.mu.data[k] = mu_k
            
            # 3. Update Lambda (Approximation via Weighted PCA)
            diff = X - mu_k
            S_k = (resp_k * diff).T @ diff / Nk[k]
            
            try:
                vals, vecs = torch.linalg.eigh(S_k)
                idx = torch.argsort(vals, descending=True)
                top_vals = vals[idx[:self.q]]
                top_vecs = vecs[:, idx[:self.q]]
                
                top_vals = torch.clamp(top_vals, min=1e-6)
                self.Lambda.data[k] = top_vecs * torch.sqrt(top_vals).unsqueeze(0)
                
                # CHANGE 3: Update Psi for this specific component!
                # Psi_k is the diagonal of the residual covariance: diag(S_k - Lambda_k * Lambda_k^T)
                L_k_updated = self.Lambda.data[k]
                recon_cov = L_k_updated @ L_k_updated.T
                
                # Extract diagonals for the update
                diag_S_k = torch.diagonal(S_k)
                diag_recon = torch.diagonal(recon_cov)
                
                psi_update = diag_S_k - diag_recon
                psi_update = torch.clamp(psi_update, min=1e-6) # Ensure strictly positive noise
                
                self.log_psi.data[k] = torch.log(psi_update)
                
            except Exception as e:
                pass # Keep old Lambda and Psi if decomposition fails
    
    def initialize_parameters(self, X):
        """
        Initialize mu and psi using K-Means++ (via scikit-learn) for better convergence.
        """
        from sklearn.cluster import KMeans
        
        X_cpu = X.cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_cpu)
        centroids = kmeans.cluster_centers_
        
        with torch.no_grad():
            self.mu.data = torch.tensor(centroids, dtype=torch.float32).to(self.mu.device)

            for k in range(self.K):
                cluster_points = X[labels == k]
                if cluster_points.shape[0] > 1:
                    var_k = torch.var(cluster_points, dim=0) + 1e-6
                    self.log_psi.data[k] = torch.log(var_k)
                    # print(f"Cluster {k}: Variance = {var_k.mean().item():.4f}")
                else:
                    self.log_psi.data[k] = torch.log(torch.ones(self.D, device=self.device) * 1e-2)
        
    def add_component(self, new_mu, new_Lambda, new_log_psi, new_weight):
        """
        Dynamically adds a new component to the Mixture Model.
        """
        with torch.no_grad():
            # 1. Update mixing proportions (pi) so they sum to 1.0
            # Convert current log_pi to linear probabilities
            current_pi = torch.exp(self.log_pi.data)
            
            # Scale old proportions down to make room for the new component's weight
            updated_pi = current_pi * (1.0 - new_weight)
            
            # Combine and convert back to log-space
            new_pi_tensor = torch.cat([updated_pi, torch.tensor([new_weight], device=self.device)])
            new_log_pi = torch.log(new_pi_tensor + 1e-10)

            # 2. Zero-Pad the new Lambda to match the global Q_MAX
            q_new = new_Lambda.shape[2]
            if q_new < self.q:
                # Calculate how many columns of zeros we need to add
                pad_size = self.q - q_new
                
                # F.pad format for 3D tensors: (pad_last_dim_left, pad_last_dim_right, pad_2nd_dim_left, ...)
                # We pad the right side of the 3rd dimension (factors)
                new_Lambda = torch.nn.functional.pad(new_Lambda, (0, pad_size, 0, 0, 0, 0))

            # 3. Concatenate all parameters
            # PyTorch requires us to re-wrap them in nn.Parameter when shapes change
            self.log_pi = nn.Parameter(new_log_pi)
            self.mu = nn.Parameter(torch.cat([self.mu.data, new_mu], dim=0))
            self.Lambda = nn.Parameter(torch.cat([self.Lambda.data, new_Lambda], dim=0))
            self.log_psi = nn.Parameter(torch.cat([self.log_psi.data, new_log_psi], dim=0))

            # 4. Increment the internal component counter
            self.K += 1
            print(f"Model successfully updated! Total components (K) is now {self.K}")
    
    
    def update_single_component(self, k, X_update, alpha=0.5):
        """
        Performs an Online M-Step update for a single Factor Analyzer.
        alpha (learning rate): 0.0 means ignore new data, 1.0 means overwrite old data.
        """
        with torch.no_grad():
            N_update = X_update.shape[0]
            
            # 1. Calculate new mean from the shelf
            new_mu = X_update.mean(dim=0)
            
            # Blend the means
            blended_mu = (1 - alpha) * self.mu.data[k] + alpha * new_mu
            self.mu.data[k] = blended_mu
            
            # 2. Calculate new covariance from the shelf
            # Center the data using the blended mean
            diff = X_update - blended_mu
            new_cov = (diff.T @ diff) / (N_update - 1)
            
            # Reconstruct the old covariance matrix: C_k = L_k @ L_k.T + Psi_k
            old_L = self.Lambda.data[k]
            old_psi = torch.exp(self.log_psi.data[k])
            old_cov = old_L @ old_L.T + torch.diag(old_psi)
            
            # Blend the covariance matrices
            blended_cov = (1 - alpha) * old_cov + alpha * new_cov
            
            # 3. Extract updated Factors (Lambda) via Eigen Decomposition
            try:
                vals, vecs = torch.linalg.eigh(blended_cov)
                # Sort descending
                idx = torch.argsort(vals, descending=True)
                vals = vals[idx]
                vecs = vecs[:, idx]
                
                # Get the number of active factors (ignore zero-padding from dynamic Q)
                active_q = (self.Lambda.data[k].sum(dim=0) != 0).sum().item()
                if active_q == 0: active_q = self.q # Fallback
                
                top_vals = torch.clamp(vals[:active_q], min=1e-6)
                top_vecs = vecs[:, :active_q]
                
                # Calculate the new unpadded Lambda
                new_L = top_vecs * torch.sqrt(top_vals).unsqueeze(0)
                
                # Pad it back to match the global Q_MAX dimension
                pad_size = self.q - active_q
                if pad_size > 0:
                    new_L = torch.nn.functional.pad(new_L, (0, pad_size, 0, 0))
                
                self.Lambda.data[k] = new_L
                
                # 4. Update specific noise (Psi)
                # Psi is the diagonal of the residual: diag(Cov - Lambda @ Lambda^T)
                recon_cov = new_L @ new_L.T
                psi_update = torch.diagonal(blended_cov) - torch.diagonal(recon_cov)
                psi_update = torch.clamp(psi_update, min=1e-6)
                
                self.log_psi.data[k] = torch.log(psi_update)
                
            except Exception as e:
                print(f"Warning: Eigendecomposition failed for component {k}. Skipping factor update. Error: {e}")