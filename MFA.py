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