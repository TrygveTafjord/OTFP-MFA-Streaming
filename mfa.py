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
        # Initialize means centered around 0 but spread out
        self.mu = nn.Parameter(torch.randn(self.K, self.D, device=self.device) * 0.1)
        # Factor loadings (K, D, q)
        self.Lambda = nn.Parameter(torch.randn(self.K, self.D, self.q, device=self.device) * 0.1)
        # Log Diagonal noise is specific to each component (K, D)
        self.log_psi = nn.Parameter(torch.log(torch.ones(self.K, self.D, device=self.device) * 1e-2))

        # We use register_buffer so they are saved in the state_dict but aren't trainable parameters
        self.register_buffer('S0', torch.zeros(self.K, device=self.device))
        self.register_buffer('S1', torch.zeros(self.K, self.D, device=self.device))
        self.register_buffer('S_xx', torch.zeros(self.K, self.D, device=self.device))      # Diagonal only!
        self.register_buffer('S_z', torch.zeros(self.K, self.q, device=self.device))
        self.register_buffer('S_xz', torch.zeros(self.K, self.D, self.q, device=self.device))
        self.register_buffer('S_zz', torch.zeros(self.K, self.q, self.q, device=self.device))

        self.register_buffer('update_counts', torch.zeros(self.K, device=self.device))
        
    def fit(self, X):
        X = X.to(self.device)
        N = X.shape[0]

        prev_ll = -float('inf')
        with torch.no_grad():
            for i in range(self.max_iter):
                
                log_resp, log_likelihood, _ , _= self.e_step(X)
                current_ll = log_likelihood.mean()
                
                resp = torch.exp(log_resp) # (N, K)
                self.m_step(X, resp)
                
                # Convergence check
                diff = current_ll - prev_ll
                if i > 0 and abs(diff) < self.tol:
                    break
                prev_ll = current_ll
                
            self.final_ll = prev_ll * N 
        
    def e_step(self, X):
        """
        Performs the Expectation step.
        Returns:
        -Normalized responsibilitiest P(x_i) = Sum(P(X_i | w_j) pi_j)
        -Total log-likelihood h_ij = P(w_j|x_i) = (P(X_i | w_j) pi_j)/P(x_i)
        -Raw geometric fits.
        """
        log_probs, mahalanobis_dists = self.compute_distances_and_log_probs(X)
        
        log_resps = log_probs + self.log_pi.unsqueeze(0)
        log_likelihood = torch.logsumexp(log_resps, dim=1)
        log_resp_norm = log_resps - log_likelihood.unsqueeze(1)
        
        return log_resp_norm, log_likelihood, log_probs, mahalanobis_dists

    def m_step(self, X, resp):
        N, D = X.shape
        Nk = resp.sum(dim=0) + 1e-10 
        
        # 1. Update Global Mixing Weights (Pi)
        self.log_pi.data = torch.log(Nk / N)
        
        for k in range(self.K):
            resp_k = resp[:, k].unsqueeze(1) # (N, 1)
            
            L_k = self.Lambda[k]                      # (D, q)
            mu_k = self.mu[k]                         # (D,)
            psi_k = torch.exp(self.log_psi[k]) + 1e-6 # (D,)
            inv_psi = 1.0 / psi_k                     # (D,)
            
            # M = I + L^T * Psi^{-1} * L
            L_k_scaled = inv_psi.unsqueeze(1) * L_k                   
            M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled
            inv_M = torch.inverse(M)                                  
            
            # beta = M^{-1} * L^T * Psi^{-1}
            beta = inv_M @ L_k_scaled.T                               # (q, D)
            
            # First Moment: E[z | x] = beta * (X - mu)
            diff = X - mu_k                                           # (N, D)
            Ez = diff @ beta.T                                        # (N, q)
            
            # Second Moment Summation: sum(h_ij * E[zz^T]) 
            # Var(z|x) elegantly reduces to exactly inv_M
            sum_Ezz = Nk[k] * inv_M + Ez.T @ (resp_k * Ez)            # (q, q)
            
            # Augmented latent factor: Ez_tilde = [Ez, 1] 
            ones = torch.ones(N, 1, device=self.device)
            Ez_tilde = torch.cat([Ez, ones], dim=1)                   # (N, q+1)
            
            # sum_i h_{ij} E[z_tilde z_tilde^T] -> shape (q+1, q+1)
            sum_h_Ez = (resp_k * Ez).sum(dim=0, keepdim=True).T       # (q, 1)
            
            top_row = torch.cat([sum_Ezz, sum_h_Ez], dim=1)           
            bottom_row = torch.cat([sum_h_Ez.T, Nk[k].unsqueeze(0).unsqueeze(0)], dim=1) 
            sum_E_ztilde_ztildeT = torch.cat([top_row, bottom_row], dim=0) # (q+1, q+1)
            
            # sum_i h_{ij} x_i Ez_tilde^T -> shape (D, q+1)
            sum_x_Eztilde = (resp_k * X).T @ Ez_tilde                 # (D, q+1)
            
            # Lambda_tilde_new = sum_x_Eztilde @ inv(sum_E_ztilde_ztildeT)
            Lambda_tilde_new = sum_x_Eztilde @ torch.inverse(sum_E_ztilde_ztildeT)       
            
            Lambda_new = Lambda_tilde_new[:, :self.q]                 # (D, q)
            mu_new = Lambda_tilde_new[:, self.q]                      # (D,)
            
            # Psi_new = diag( sum_i h_ij (x_i x_i^T - Lambda_tilde_new E_ztilde_i x_i^T) ) / Nk[k]
            diag_xx = (resp_k * X * X).sum(dim=0)                     # (D,)
            
            # The cross term simplifies to row-wise multiplication, avoiding O(D^2) memory allocations
            diag_cross = (Lambda_tilde_new * sum_x_Eztilde).sum(dim=1) # (D,)
            
            psi_update = (diag_xx - diag_cross) / Nk[k]
            psi_update = torch.clamp(psi_update, min=1e-5)
            
            self.Lambda.data[k] = Lambda_new
            self.mu.data[k] = mu_new
            self.log_psi.data[k] = torch.log(psi_update)


    def stepwise_em_update(self, X, log_resp):
        X = X.to(self.device)
        N = X.shape[0]
        resp = torch.exp(log_resp) # (N, K)

        # 1. Batch Responsibilities
        s0_batch = resp.sum(dim=0) / N                             # (K,)
        s1_batch = (resp.T @ X) / N                                # (K, D)
        s_xx_batch = (resp.T @ (X ** 2)) / N                       # (K, D) - Diagonal only!

        for k in range(self.K):
            if s0_batch[k] < 1e-6:
                continue
            
            resp_k = resp[:, k].unsqueeze(1) # (N, 1)

            # --- E-STEP for Latent Variables ---
            L_k = self.Lambda[k]                      
            mu_k = self.mu[k]                         
            inv_psi = 1.0 / (torch.exp(self.log_psi[k]) + 1e-6) 

            L_k_scaled = inv_psi.unsqueeze(1) * L_k                   
            M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled
            inv_M = torch.inverse(M)                                  
            beta = inv_M @ L_k_scaled.T                               

            diff = X - mu_k                                           
            Ez = diff @ beta.T                                        # E[z|x] -> (N, q)

            # --- Calculate Batch Sufficient Statistics ---
            sz_batch = (resp_k * Ez).sum(dim=0) / N                   # (q,)
            sxz_batch = ((resp_k * X).T @ Ez) / N                     # (D, q)

            # E[zz^T|x] summation
            sum_Ezz = (s0_batch[k] * N) * inv_M + Ez.T @ (resp_k * Ez)
            szz_batch = sum_Ezz / N                                   # (q, q)

            # --- Interpolate Global Sufficient Statistics ---
            k_count = self.update_counts[k].item()
            eta = (k_count + 2) ** (-self.alpha)

            if k_count == 0:
                self.S0[k] = s0_batch[k]
                self.S1[k] = s1_batch[k]
                self.S_xx[k] = s_xx_batch[k]
                self.S_z[k] = sz_batch
                self.S_xz[k] = sxz_batch
                self.S_zz[k] = szz_batch
            else:
                self.S0[k] = (1 - eta) * self.S0[k] + eta * s0_batch[k]
                self.S1[k] = (1 - eta) * self.S1[k] + eta * s1_batch[k]
                self.S_xx[k] = (1 - eta) * self.S_xx[k] + eta * s_xx_batch[k]
                self.S_z[k] = (1 - eta) * self.S_z[k] + eta * sz_batch
                self.S_xz[k] = (1 - eta) * self.S_xz[k] + eta * sxz_batch
                self.S_zz[k] = (1 - eta) * self.S_zz[k] + eta * szz_batch

            self.update_counts[k] += 1

            # --- M-STEP: Recover Parameters ---
            self.log_pi.data = torch.log(self.S0 / self.S0.sum() + 1e-10)

            mu_k_new = self.S1[k] / (self.S0[k] + 1e-10)
            self.mu.data[k] = mu_k_new

            # Exact Lambda update: (S_xz - mu * S_z^T) * S_zz^-1
            S_z_scaled = self.S_z[k].unsqueeze(0) # (1, q)
            mu_k_col = mu_k_new.unsqueeze(1)      # (D, 1)

            lambda_num = self.S_xz[k] - (mu_k_col @ S_z_scaled)
            Lambda_new = lambda_num @ torch.inverse(self.S_zz[k] + torch.eye(self.q, device=self.device)*1e-6)
            self.Lambda.data[k] = Lambda_new

            # Exact Psi update (diagonal only)
            diag_cross = (Lambda_new * lambda_num).sum(dim=1)

            # Add a ridge penalty (Tikhonov regularization) to prevent variance collapse
            ridge_penalty = 1e-6 

            psi_update = (self.S_xx[k] - (mu_k_new * self.S1[k]) - diag_cross + ridge_penalty) / (self.S0[k] + 1e-10)

            # You can keep the clamp as an absolute fail-safe
            self.log_psi.data[k] = torch.log(torch.clamp(psi_update, min=1e-5))
    
    def compute_distances_and_log_probs(self, X):
        """
        Calculates the pure structural/geometric fit using the Woodbury Matrix Identity
        to avoid O(D^3) inversion of the 120x120 covariance matrix.
        This calculates log P(x_i | w_j) and entirely ignores the global mixing weights (pi).
        """
        N, D = X.shape
        log_probs = []
        mahalanobis_dists = []
        
        # Constant term for the log pdf
        c = D * math.log(2 * math.pi)
        
        for k in range(self.K):
            L_k = self.Lambda[k]                     # (D, q)
            psi_k = torch.exp(self.log_psi[k]) + 1e-6 # (D,)
            inv_psi = 1.0 / psi_k                    # (D,)
            
            # 1. Calculate the q x q inner matrix: M = I + Lambda^T * Psi^{-1} * Lambda
            # Since Psi is diagonal, Psi^{-1} * Lambda is just row-wise multiplication
            L_k_scaled = inv_psi.unsqueeze(1) * L_k  # (D, q)
            M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled # (q, q)
            
            # Invert the tiny q x q matrix
            inv_M = torch.inverse(M)                 # (q, q)
            
            # 2. Calculate the Log Determinant using the determinant lemma:
            # |C| = |Psi| * |M|
            log_det_psi = torch.sum(torch.log(psi_k))
            log_det_M = torch.logdet(M)
            log_det_C = log_det_psi + log_det_M
            
            # 3. Calculate Mahalanobis distance: (x-mu)^T C^{-1} (x-mu)
            diff = X - self.mu[k]                    # (N, D)
            
            # Term 1: diff^T * Psi^{-1} * diff
            term1 = torch.sum((diff ** 2) * inv_psi, dim=1) # (N,)
            
            # Term 2: diff^T * Psi^{-1} * Lambda * M^{-1} * Lambda^T * Psi^{-1} * diff
            # Let proj = diff * Psi^{-1} * Lambda
            proj = diff @ L_k_scaled                 # (N, q)
            term2 = torch.sum((proj @ inv_M) * proj, dim=1) # (N,)
            
            mahalanobis = term1 - term2              # (N,)
            mahalanobis_dists.append(mahalanobis)

            # 4. Final Log PDF
            log_prob = -0.5 * (c + log_det_C + mahalanobis)
            log_probs.append(log_prob)

            # Mahalanobis distances can be useful for OTFP drift detection, so we store them as well
                
        return torch.stack(log_probs, dim=1), torch.stack(mahalanobis_dists, dim=1) # NEW: Return both    
        
    def add_component(self, X_pure, total_samples_seen, new_mu, new_Lambda, new_log_psi):
        """
        Dynamically adds a new component to the Mixture Model.
        Calculates and integrates its sufficient statistics internally.
        """
        with torch.no_grad():
            N_pure = X_pure.shape[0]
            q_new = new_Lambda.shape[2]
            
            # 1. Calculate Scaled Sufficient Statistics Internally
            new_S0 = torch.tensor([N_pure / max(total_samples_seen, 1)], dtype=torch.float32, device=self.device)
            new_S1 = new_S0 * X_pure.mean(dim=0, keepdim=True)            
            new_S_xx = new_S0 * (X_pure ** 2).mean(dim=0, keepdim=True) 
            
            new_S_z = torch.zeros((1, q_new), device=self.device)
            new_S_xz = torch.zeros((1, self.D, q_new), device=self.device)
            new_S_zz = new_S0 * torch.eye(q_new, device=self.device).unsqueeze(0) 

            # 2. Handle Tensor Shape Matching (Padding Lambda and Latent Statistics)
            if q_new < self.q:
                pad_size = self.q - q_new
                new_Lambda = torch.nn.functional.pad(new_Lambda, (0, pad_size))
                new_S_z = torch.nn.functional.pad(new_S_z, (0, pad_size))
                new_S_xz = torch.nn.functional.pad(new_S_xz, (0, pad_size))
                new_S_zz = torch.nn.functional.pad(new_S_zz, (0, pad_size, 0, pad_size))
                
            elif q_new > self.q:
                pad_size = q_new - self.q
                self.Lambda.data = torch.nn.functional.pad(self.Lambda.data, (0, pad_size))
                self.S_z = torch.nn.functional.pad(self.S_z, (0, pad_size))
                self.S_xz = torch.nn.functional.pad(self.S_xz, (0, pad_size))
                self.S_zz = torch.nn.functional.pad(self.S_zz, (0, pad_size, 0, pad_size))
                
                self.q = q_new

            # 3. Concatenate the standard parameters safely
            self.mu = nn.Parameter(torch.cat([self.mu.data, new_mu], dim=0))
            self.Lambda = nn.Parameter(torch.cat([self.Lambda.data, new_Lambda], dim=0))
            self.log_psi = nn.Parameter(torch.cat([self.log_psi.data, new_log_psi], dim=0))

            # 4. Concatenate the Sufficient Statistics
            self.S0 = torch.cat([self.S0, new_S0])
            self.S1 = torch.cat([self.S1, new_S1])
            self.S_xx = torch.cat([self.S_xx, new_S_xx])
            self.S_z = torch.cat([self.S_z, new_S_z])
            self.S_xz = torch.cat([self.S_xz, new_S_xz])
            self.S_zz = torch.cat([self.S_zz, new_S_zz])
            
            # 5. Initialize the update count to 1
            self.update_counts = torch.cat([self.update_counts, torch.tensor([1.0], device=self.device)])

            # 6. Automatically update global mixing weights (log_pi)
            self.log_pi = nn.Parameter(torch.log(self.S0 / self.S0.sum() + 1e-10))

            self.K += 1
            print(f"Model successfully updated! Total components (K) is now {self.K}")

    def init_sufficient_statistics(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            log_resp_norm, _, _, _ = self.e_step(X)
            resp = torch.exp(log_resp_norm) 
            N = X.shape[0]
            
            # Divide by N so these represent normalized proportions!
            self.S0 = resp.sum(dim=0) / N 
            self.S1 = (resp.T @ X) / N      
            self.S_xx = (resp.T @ (X ** 2)) / N  # Tracking the diagonal squared observations
            
            for k in range(self.K):
                if self.S0[k] < 1e-6:
                    continue # Skip empty/dead components
                    
                resp_k = resp[:, k].unsqueeze(1) # (N, 1)
                
                # --- E-STEP to recover latent expectations for initialization ---
                L_k = self.Lambda[k]                      
                mu_k = self.mu[k]                         
                inv_psi = 1.0 / (torch.exp(self.log_psi[k]) + 1e-6) 
                
                L_k_scaled = inv_psi.unsqueeze(1) * L_k                   
                M = torch.eye(self.q, device=self.device) + L_k.T @ L_k_scaled
                inv_M = torch.inverse(M)                                  
                beta = inv_M @ L_k_scaled.T                               
                
                diff = X - mu_k                                           
                Ez = diff @ beta.T                                        # E[z|x] -> (N, q)
                
                # --- Initialize Latent Sufficient Statistics ---
                self.S_z[k] = (resp_k * Ez).sum(dim=0) / N                   # (q,)
                self.S_xz[k] = ((resp_k * X).T @ Ez) / N                     # (D, q)
                
                # E[zz^T|x] summation
                sum_Ezz = (self.S0[k] * N) * inv_M + Ez.T @ (resp_k * Ez)
                self.S_zz[k] = sum_Ezz / N                                   # (q, q)
            
            self.update_counts = torch.ones(self.K, device=self.device)
    
    
    