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
        self.register_buffer('S2', torch.zeros(self.K, self.D, self.D, device=self.device))
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
        
        # NEW: Return the distances alongside your existing outputs
        return log_resp_norm, log_likelihood, log_probs, mahalanobis_dists

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
                psi_update = torch.clamp(psi_update, min=1e-5) # Ensure strictly positive noise
                
                self.log_psi.data[k] = torch.log(psi_update)
                
            except Exception as e:
                pass # Keep old Lambda and Psi if decomposition fails

    def stepwise_em_update(self, X, log_resp):
        """
        Performs a single online Stepwise EM update using a mini-batch of streaming data.
        """
        X = X.to(self.device)
        N = X.shape[0]
        resp = torch.exp(log_resp) # (N, K)
        
        # 1. Calculate Batch Sufficient Statistics
        s0_batch = resp.sum(dim=0) / N                             # (K,)
        s1_batch = (resp.T @ X) / N                                # (K, D)
        
        for k in range(self.K):
            if s0_batch[k] < 1e-6:
                continue # Skip if component has virtually no responsibility in this batch
            
            # Efficiently calculate batch S2 (D x D) without explicit N x D x D outer products
            resp_k = resp[:, k:k+1] # (N, 1)
            s2_batch_k = ((resp_k * X).T @ X) / N                  # (D, D)
            
            # 2. Adaptive Learning Rate (eta) based on component maturity
            k_count = self.update_counts[k].item()
            eta = (k_count + 2) ** (-self.alpha)
            
            # 3. Interpolate Global Sufficient Statistics
            if k_count == 0:
                # First time seeing data, accept batch stats completely
                self.S0[k] = s0_batch[k]
                self.S1[k] = s1_batch[k]
                self.S2[k] = s2_batch_k
            else:
                self.S0[k] = (1 - eta) * self.S0[k] + eta * s0_batch[k]
                self.S1[k] = (1 - eta) * self.S1[k] + eta * s1_batch[k]
                self.S2[k] = (1 - eta) * self.S2[k] + eta * s2_batch_k
            
            self.update_counts[k] += 1
            
            # 4. M-Step: Recover Parameters from Global Statistics
            # Update Pi (Global mix proxy)
            self.log_pi.data = torch.log(self.S0 / self.S0.sum() + 1e-10)
            
            # Update Mu
            mu_k = self.S1[k] / (self.S0[k] + 1e-10)
            self.mu.data[k] = mu_k
            
            # Update Covariance, Lambda, and Psi
            # Sigma_k = S2 / S0 - (mu * mu^T)
            Sigma_k = (self.S2[k] / (self.S0[k] + 1e-10)) - torch.outer(mu_k, mu_k)
            
            try:
                # Weighted PCA for Factor Loadings
                vals, vecs = torch.linalg.eigh(Sigma_k)
                idx = torch.argsort(vals, descending=True)
                top_vals = torch.clamp(vals[idx[:self.q]], min=1e-6)
                top_vecs = vecs[:, idx[:self.q]]
                
                L_k_updated = top_vecs * torch.sqrt(top_vals).unsqueeze(0)
                self.Lambda.data[k] = L_k_updated
                
                # Update Psi (Diagonal noise)
                recon_cov = L_k_updated @ L_k_updated.T
                psi_update = torch.diagonal(Sigma_k) - torch.diagonal(recon_cov)
                psi_update = torch.clamp(psi_update, min=1e-5)
                self.log_psi.data[k] = torch.log(psi_update)
                
            except Exception as e:
                # Fallback if decomposition fails due to numerical instability
                pass
    
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
    
        
    def add_component(self, new_mu, new_Lambda, new_log_psi, new_S0, new_S1, new_S2):
        """
        Dynamically adds a new component to the Mixture Model and integrates its sufficient statistics.
        """
        with torch.no_grad():
            # 1. Handle Tensor Shape Matching (Padding Lambda)
            q_new = new_Lambda.shape[2]
            
            if q_new < self.q:
                pad_size = self.q - q_new
                new_Lambda = torch.nn.functional.pad(new_Lambda, (0, pad_size))
            elif q_new > self.q:
                pad_size = q_new - self.q
                self.Lambda.data = torch.nn.functional.pad(self.Lambda.data, (0, pad_size))
                self.q = q_new

            # 2. Concatenate the standard parameters safely
            self.mu = nn.Parameter(torch.cat([self.mu.data, new_mu], dim=0))
            self.Lambda = nn.Parameter(torch.cat([self.Lambda.data, new_Lambda], dim=0))
            self.log_psi = nn.Parameter(torch.cat([self.log_psi.data, new_log_psi], dim=0))

            # 3. Concatenate the Sufficient Statistics
            self.S0 = torch.cat([self.S0, new_S0])
            self.S1 = torch.cat([self.S1, new_S1])
            self.S2 = torch.cat([self.S2, new_S2])
            
            # 4. Initialize the update count to 1 (it has seen its foundational batch)
            self.update_counts = torch.cat([self.update_counts, torch.tensor([1.0], device=self.device)])

            # 5. Automatically update global mixing weights (log_pi) based on the new total S0 mass
            self.log_pi = nn.Parameter(torch.log(self.S0 / self.S0.sum() + 1e-10))

            # Increment the internal component counter
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
            
            for k in range(self.K):
                resp_k = resp[:, k:k+1] 
                self.S2[k] = (resp_k * X).T @ X / N 
            
            self.update_counts = torch.ones(self.K, device=self.device)
    
    
    