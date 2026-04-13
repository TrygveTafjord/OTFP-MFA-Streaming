import torch
import torch.nn as nn
import math

class MFA(nn.Module):
    def __init__(self, n_components, n_channels, n_factors, tol=1e-4, max_iter=150, device='cpu', alpha=0.6):
        super().__init__()
        self.K = n_components
        self.D = n_channels
        self.q = n_factors
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        self.alpha = alpha
        
        # Initialize parameters
        self.log_pi = nn.Parameter(torch.log(torch.ones(self.K, device=self.device) / self.K))
        self.mu = nn.Parameter(torch.randn(self.K, self.D, device=self.device) * 0.1)
        self.Lambda = nn.Parameter(torch.randn(self.K, self.D, self.q, device=self.device) * 0.1)
        self.log_psi = nn.Parameter(torch.log(torch.ones(self.K, self.D, device=self.device) * 1e-2))

        # --- MEMORY EFFICIENT SUFFICIENT STATISTICS ---
        # We replace the DxD S2 matrix with theoretically pure latent statistics.
        self.register_buffer('S0', torch.zeros(self.K, device=self.device))                  # 0th order (Scalar)
        self.register_buffer('S1', torch.zeros(self.K, self.D, device=self.device))          # 1st order Y (D)
        self.register_buffer('SX', torch.zeros(self.K, self.q, device=self.device))          # 1st order Latent X (q)
        self.register_buffer('SXX', torch.zeros(self.K, self.q, self.q, device=self.device)) # 2nd order Latent XX (q x q)
        self.register_buffer('SYX', torch.zeros(self.K, self.D, self.q, device=self.device)) # Cross-order YX (D x q)
        self.register_buffer('SY2', torch.zeros(self.K, self.D, device=self.device))         # Diagonal of YY (D)
        
        self.register_buffer('update_counts', torch.zeros(self.K, device=self.device))
        
    def fit(self, X):
        X = X.to(self.device)
        N = X.shape[0]

        prev_ll = -float('inf')
        with torch.no_grad():
            for i in range(self.max_iter):
                log_resp, log_likelihood, _ = self.e_step(X)
                current_ll = log_likelihood.mean()
                
                resp = torch.exp(log_resp) # (N, K)
                self.m_step(X, resp)
                
                diff = current_ll - prev_ll
                if i > 0 and abs(diff) < self.tol:
                    break
                prev_ll = current_ll
                
            self.final_ll = prev_ll * N 
        
    def e_step(self, X):
        log_probs = self.compute_component_log_likelihoods(X)
        log_resps = log_probs + self.log_pi.unsqueeze(0)
        log_likelihood = torch.logsumexp(log_resps, dim=1)
        log_resp_norm = log_resps - log_likelihood.unsqueeze(1)
        return log_resp_norm, log_likelihood, log_probs

    def m_step(self, X, resp):
        """
        Batch M-Step. Refactored to use the exact sufficient statistics math 
        instead of the unstable PCA approximation.
        """
        N = X.shape[0]
        Nk = resp.sum(dim=0) + 1e-10 
        self.log_pi.data = torch.log(Nk / N)
        
        for k in range(self.K):
            resp_k = resp[:, k:k+1] 
            
            # 1. Woodbury Identity Latent Moments [cite: 706, 728]
            L_k = self.Lambda[k]                     
            psi_k = torch.exp(self.log_psi[k]) + 1e-6 
            inv_psi = 1.0 / psi_k                    
            
            L_k_scaled = inv_psi.unsqueeze(1) * L_k  
            M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled 
            M_inv = torch.inverse(M)                 
            
            A_k = M_inv @ L_k_scaled.T               
            diff = X - self.mu[k]                    
            
            # E[x] and V[x] computation based on Eq 22 and 23 [cite: 728, 730]
            E_k = diff @ A_k.T                       
            
            # 2. Local Batch Statistics
            S0_k = Nk[k]
            S1_k = (resp_k * X).sum(dim=0)
            SX_k = (resp_k * E_k).sum(dim=0)
            SXX_k = S0_k * M_inv + (resp_k * E_k).T @ E_k 
            SYX_k = (resp_k * X).T @ E_k
            SY2_k = (resp_k * X**2).sum(dim=0)
            
            # 3. Exact Parameter Updates based on Appendix A Eqs 26-28 [cite: 708, 709]
            mu_new = S1_k / S0_k
            self.mu.data[k] = mu_new
            
            # Eq 26: Lambda_new = C * SXX^-1 [cite: 708]
            C = (SYX_k - torch.outer(mu_new, SX_k)) / S0_k 
            SXX_norm = SXX_k / S0_k
            Lambda_new = C @ torch.inverse(SXX_norm)
            self.Lambda.data[k] = Lambda_new
            
            # Eq 28: Extracting purely the diagonal noise update [cite: 708, 709]
            diag_Lambda_C = torch.sum(Lambda_new * C, dim=1)
            psi_new = (SY2_k / S0_k) - mu_new**2 - diag_Lambda_C
            
            self.log_psi.data[k] = torch.log(torch.clamp(psi_new, min=1e-6))

    def stepwise_em_update(self, X, log_resp):
        """
        Pure Online Stepwise EM based on Cappé & Moulines sufficient statistics framework.
        Tracks moments using learning rate eta to adapt to concept drift.
        """
        X = X.to(self.device)
        N = X.shape[0]
        resp = torch.exp(log_resp) 
        
        for k in range(self.K):
            resp_k = resp[:, k:k+1] 
            sum_resp = resp_k.sum()
            
            if sum_resp < 1e-6:
                continue 
                
            # --- Latent Moments via Woodbury (E-Step) ---
            L_k = self.Lambda[k]                     
            psi_k = torch.exp(self.log_psi[k]) + 1e-6 
            inv_psi = 1.0 / psi_k                    
            
            L_k_scaled = inv_psi.unsqueeze(1) * L_k  
            M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled 
            M_inv = torch.inverse(M)                 
            
            A_k = M_inv @ L_k_scaled.T               
            diff = X - self.mu[k]                    
            
            # <x> = A_k(y - mu) [cite: 728]
            E_k = diff @ A_k.T                       
            
            # --- Batch Sufficient Statistics ---
            s0_batch = sum_resp / N
            s1_batch = (resp_k * X).sum(dim=0) / N
            sX_batch = (resp_k * E_k).sum(dim=0) / N
            
            # <xx^T> = M^-1 + <x><x^T> [cite: 730]
            sXX_batch = (s0_batch * M_inv) + ((resp_k * E_k).T @ E_k) / N
            sYX_batch = (resp_k * X).T @ E_k / N
            sY2_batch = (resp_k * X ** 2).sum(dim=0) / N
            
            # --- Adaptive Learning Rate ---
            k_count = self.update_counts[k].item()
            eta = (k_count + 2) ** (-self.alpha)
            
            # --- Interpolate Global Statistics ---
            if k_count == 0:
                self.S0[k] = s0_batch
                self.S1[k] = s1_batch
                self.SX[k] = sX_batch
                self.SXX[k] = sXX_batch
                self.SYX[k] = sYX_batch
                self.SY2[k] = sY2_batch
            else:
                self.S0[k] = (1 - eta) * self.S0[k] + eta * s0_batch
                self.S1[k] = (1 - eta) * self.S1[k] + eta * s1_batch
                self.SX[k] = (1 - eta) * self.SX[k] + eta * sX_batch
                self.SXX[k] = (1 - eta) * self.SXX[k] + eta * sXX_batch
                self.SYX[k] = (1 - eta) * self.SYX[k] + eta * sYX_batch
                self.SY2[k] = (1 - eta) * self.SY2[k] + eta * sY2_batch
                
            self.update_counts[k] += 1
            
            # --- M-Step (Parameter Updates from global buffers) ---
            S0_k = self.S0[k] + 1e-10
            
            mu_new = self.S1[k] / S0_k
            self.mu.data[k] = mu_new
            
            C = (self.SYX[k] - torch.outer(mu_new, self.SX[k])) / S0_k
            SXX_norm = self.SXX[k] / S0_k
            
            Lambda_new = C @ torch.inverse(SXX_norm)
            self.Lambda.data[k] = Lambda_new
            
            diag_Lambda_C = torch.sum(Lambda_new * C, dim=1)
            psi_new = (self.SY2[k] / S0_k) - mu_new**2 - diag_Lambda_C
            
            self.log_psi.data[k] = torch.log(torch.clamp(psi_new, min=1e-6))
            
        self.log_pi.data = torch.log(self.S0 / self.S0.sum() + 1e-10)
    
    def compute_component_log_likelihoods(self, X):
        N, D = X.shape
        log_probs = []
        c = D * math.log(2 * math.pi)
        
        for k in range(self.K):
            L_k = self.Lambda[k]                     
            psi_k = torch.exp(self.log_psi[k]) + 1e-6 
            inv_psi = 1.0 / psi_k                    
            
            L_k_scaled = inv_psi.unsqueeze(1) * L_k  
            M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled 
            inv_M = torch.inverse(M)                 
            
            log_det_psi = torch.sum(torch.log(psi_k))
            log_det_M = torch.logdet(M)
            log_det_C = log_det_psi + log_det_M
            
            diff = X - self.mu[k]                    
            term1 = torch.sum((diff ** 2) * inv_psi, dim=1) 
            
            proj = diff @ L_k_scaled                 
            term2 = torch.sum((proj @ inv_M) * proj, dim=1) 
            
            mahalanobis = term1 - term2              
            log_prob = -0.5 * (c + log_det_C + mahalanobis)
            log_probs.append(log_prob)
            
        return torch.stack(log_probs, dim=1)
    
    def initialize_parameters(self, X):
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
                else:
                    self.log_psi.data[k] = torch.log(torch.ones(self.D, device=self.device) * 1e-2)
        
    def add_component(self, new_mu, new_Lambda, new_log_psi, new_S0, new_S1, new_S2):
        """
        Dynamically adds a component. 
        We mathematically translate the DxD S2 matrix coming from otfp.py 
        into our memory-efficient latent sufficient statistics. 
        """
        with torch.no_grad():
            q_new = new_Lambda.shape[2]
            if q_new < self.q:
                pad_size = self.q - q_new
                new_Lambda = torch.nn.functional.pad(new_Lambda, (0, pad_size))
            elif q_new > self.q:
                pad_size = q_new - self.q
                self.Lambda.data = torch.nn.functional.pad(self.Lambda.data, (0, pad_size))
                
                # Expand existing statistics buffers along the q dimension
                self.SX = torch.nn.functional.pad(self.SX, (0, pad_size))
                self.SXX = torch.nn.functional.pad(self.SXX, (0, pad_size, 0, pad_size))
                self.SYX = torch.nn.functional.pad(self.SYX, (0, pad_size))
                self.q = q_new

            self.mu = nn.Parameter(torch.cat([self.mu.data, new_mu], dim=0))
            self.Lambda = nn.Parameter(torch.cat([self.Lambda.data, new_Lambda], dim=0))
            self.log_psi = nn.Parameter(torch.cat([self.log_psi.data, new_log_psi], dim=0))

           # --- Translate DxD external input into Latent Statistics ---
            S0_in = new_S0[0]
            S1_in = new_S1[0]
            S2_in = new_S2[0] # D x D 

            # FIX: We must normalize these inputs so they match the scale 
            # of our global moving average buffers (which are expectations per pixel).
            # If S0_in is already normalized (e.g., <= 1.0), this division is safe.
            # If S0_in is an unnormalized sum (e.g., 2000), this converts it.
            N_shelf = S0_in.clone() 
            S0_norm = S0_in / N_shelf  # Will equal 1.0 (100% of its own shelf mass)
            S1_norm = S1_in / N_shelf
            S2_norm = S2_in / N_shelf

            L = new_Lambda[0] 
            psi = torch.exp(new_log_psi[0]) + 1e-6
            inv_psi = 1.0 / psi
            L_scaled = inv_psi.unsqueeze(1) * L
            M = torch.eye(self.q, device=self.device) + L.T @ L_scaled
            M_inv = torch.inverse(M)
            A = M_inv @ L_scaled.T 

            # Use the normalized values for the translations
            mu_in = S1_norm / S0_norm 

            new_SX = (A @ (S1_norm - S0_norm * mu_in)).unsqueeze(0)
            new_SYX = ((S2_norm - torch.outer(S1_norm, mu_in)) @ A.T).unsqueeze(0)
            cov_Y = S2_norm - torch.outer(S1_norm, mu_in) - torch.outer(mu_in, S1_norm) + S0_norm * torch.outer(mu_in, mu_in)
            new_SXX = (S0_norm * M_inv + A @ cov_Y @ A.T).unsqueeze(0)
            new_SY2 = torch.diagonal(S2_norm).unsqueeze(0)

            # NOTE: For S0, we append S0_norm (which is 1.0). But when calculating pi, 
            # we want the initial weight to be proportional. In an online setting, 
            # it is safe to initialize a new component's S0 as the average of the others.
            avg_S0 = self.S0.mean().unsqueeze(0) if self.K > 0 else torch.tensor([1.0], device=self.device)

            self.S0 = torch.cat([self.S0, avg_S0])
            self.S1 = torch.cat([self.S1, S1_norm.unsqueeze(0) * avg_S0]) # Scale S1 to match new S0
            self.SX = torch.cat([self.SX, new_SX * avg_S0])
            self.SXX = torch.cat([self.SXX, new_SXX * avg_S0])
            self.SYX = torch.cat([self.SYX, new_SYX * avg_S0])
            self.SY2 = torch.cat([self.SY2, new_SY2 * avg_S0])
            
            self.update_counts = torch.cat([self.update_counts, torch.tensor([1.0], device=self.device)])
            self.log_pi = nn.Parameter(torch.log(self.S0 / self.S0.sum() + 1e-10))

            self.K += 1
            print(f"Model successfully updated! Total components (K) is now {self.K}")
    
    def init_sufficient_statistics(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            log_resp_norm, _, _ = self.e_step(X)
            resp = torch.exp(log_resp_norm) 
            N = X.shape[0]
            
            self.S0 = resp.sum(dim=0) / N 
            self.S1 = (resp.T @ X) / N      
            
            for k in range(self.K):
                resp_k = resp[:, k:k+1] 
                
                L_k = self.Lambda[k]
                psi_k = torch.exp(self.log_psi[k]) + 1e-6
                inv_psi = 1.0 / psi_k
                L_k_scaled = inv_psi.unsqueeze(1) * L_k
                M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled
                M_inv = torch.inverse(M)
                A_k = M_inv @ L_k_scaled.T
                
                diff = X - self.mu[k]
                E_k = diff @ A_k.T
                
                self.SX[k] = (resp_k * E_k).sum(dim=0) / N
                self.SXX[k] = (self.S0[k] * M_inv) + ((resp_k * E_k).T @ E_k) / N
                self.SYX[k] = (resp_k * X).T @ E_k / N
                self.SY2[k] = (resp_k * X**2).sum(dim=0) / N
            
            self.update_counts = torch.ones(self.K, device=self.device)