import torch
import torch.nn as nn
import math

class MFA(nn.Module):
    def __init__(self, n_components, n_channels, n_factors, tol=1e-4, max_iter=150, device='cpu', alpha=0.5):
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
        self.register_buffer('q_k', torch.ones(self.K, dtype=torch.long, device=self.device) * self.q)
        
    def fit(self, X):
        X = X.to(self.device)
        N = X.shape[0]

        prev_ll = -float('inf')
        with torch.no_grad():
            for i in range(self.max_iter):
                # --- E-Step ---
                log_resp, log_likelihood, _ = self.e_step(X)
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
        """
        Performs the Expectation step.
        Returns:
        -Normalized responsibilitiest P(x_i) = Sum(P(X_i | w_j) pi_j)
        -Total log-likelihood h_ij = P(w_j|x_i) = (P(X_i | w_j) pi_j)/P(x_i)
        -Raw geometric fits.
        """
        # 1. Get the pure physical/geometric fit (Shape: [N, K])
        log_probs = self.compute_component_log_likelihoods(X)
        
        # 2. Add the Bayesian Prior (log_pi) to get unnormalized responsibilities
        # self.log_pi has shape [K]. unsqueeze(0) makes it [1, K] for perfect broadcasting against [N, K]
        log_resps = log_probs + self.log_pi.unsqueeze(0)
        
        # 3. Marginalize to find the total log-likelihood for the Purity Guardrail (Shape: [N])
        log_likelihood = torch.logsumexp(log_resps, dim=1)
        
        # 4. Normalize to get the final posterior responsibilities (Shape: [N, K])
        log_resp_norm = log_resps - log_likelihood.unsqueeze(1)
        
        # Return all three so OTFP can use geometry for drift, and EM can use resps for updates
        return log_resp_norm, log_likelihood, log_probs

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
                
                # Sort the values and vectors
                sorted_vals = vals[idx]
                sorted_vecs = vecs[:, idx]
                
                # ==========================================================
                # THE ELASTIC MASKING LOGIC
                # ==========================================================
                # 1. Calculate the current variance structure of the component
                total_variance = torch.sum(torch.clamp(sorted_vals, min=0.0))
                
                if total_variance > 0:
                    cumulative_variance = torch.cumsum(torch.clamp(sorted_vals, min=0.0), dim=0) / total_variance
                    
                    # 2. Find how many factors are currently needed to explain 95% of variance
                    needed_q = torch.searchsorted(cumulative_variance, 0.95).item() + 1
                    
                    # 3. Cap it at the global hardware limit (self.q)
                    needed_q = min(needed_q, self.q)
                    
                    # Optional: Print when a component actively changes its dimensionality online
                    if needed_q != self.q_k.data[k]:
                        print(f"Streaming Drift: Component {k} adapted dimensionality from {int(self.q_k.data[k].item())} to {needed_q}")
                        
                    # 4. Update the component's specific tracker
                    self.q_k.data[k] = needed_q
                
                local_q = int(self.q_k.data[k].item())
                # ==========================================================
                
                top_vals = torch.clamp(sorted_vals[:self.q], min=1e-6)
                top_vecs = sorted_vecs[:, :self.q]
                
                # Apply the mask based on the newly calculated local_q
                if local_q < self.q:
                    top_vecs[:, local_q:] = 0.0
                    top_vals[local_q:] = 1e-6 
                
                L_k_updated = top_vecs * torch.sqrt(top_vals).unsqueeze(0)
                self.Lambda.data[k] = L_k_updated
                
                recon_cov = L_k_updated @ L_k_updated.T
                psi_update = torch.diagonal(S_k) - torch.diagonal(recon_cov)
                psi_update = torch.clamp(psi_update, min=1e-3)
                self.log_psi.data[k] = torch.log(psi_update)
                
            except Exception as e:
                pass

    def stepwise_em_update(self, X, log_resp):
        X = X.to(self.device)
        N = X.shape[0]
        resp = torch.exp(log_resp) 
        
        # 1. Calculate Batch Sufficient Statistics (Expectations)
        s0_batch = resp.mean(dim=0)                             
        s1_batch = (resp.T @ X) / N                             
        
        for k in range(self.K):
            k_count = self.update_counts[k].item()
            eta = (k_count + 2) ** (-self.alpha)

            # If the component is missing from this batch, decay its history evenly
            if s0_batch[k] < 1e-6:
                self.S0[k] = (1 - eta) * self.S0[k]
                self.S1[k] = (1 - eta) * self.S1[k]
                self.S2[k] = (1 - eta) * self.S2[k]
                continue 
            
            # Efficient S2 calculation
            resp_k = resp[:, k:k+1] 
            s2_batch_k = ((resp_k * X).T @ X) / N                  
            
            # 3. Interpolate Global Sufficient Statistics
            if k_count == 0:
                self.S0[k] = s0_batch[k]
                self.S1[k] = s1_batch[k]
                self.S2[k] = s2_batch_k
            else:
                self.S0[k] = (1 - eta) * self.S0[k] + eta * s0_batch[k]
                self.S1[k] = (1 - eta) * self.S1[k] + eta * s1_batch[k]
                self.S2[k] = (1 - eta) * self.S2[k] + eta * s2_batch_k
            
            self.update_counts[k] += 1
            
            # 4. M-Step: Recover Parameters from Global Statistics
            mu_k = self.S1[k] / (self.S0[k] + 1e-10)
            self.mu.data[k] = mu_k
            
            # Calculate Covariance and enforce strict numerical symmetry
            Sigma_k = (self.S2[k] / (self.S0[k] + 1e-10)) - torch.outer(mu_k, mu_k)
            Sigma_k = 0.5 * (Sigma_k + Sigma_k.T) 
            
            try:
                vals, vecs = torch.linalg.eigh(Sigma_k)
                idx = torch.argsort(vals, descending=True)
                
                top_vals = torch.clamp(vals[idx[:self.q]], min=1e-6)
                top_vecs = vecs[:, idx[:self.q]]
                
                # === THE NEW MASKING LOGIC ===
                # Use .data to bypass Pylance "Module" warning
                local_q = int(self.q_k.data[k].item()) 
                if local_q < self.q:
                    # Zero out the unused factors
                    top_vecs[:, local_q:] = 0.0
                    # Set their variance contribution to the noise floor
                    top_vals[local_q:] = 1e-6 
                
                L_k_updated = top_vecs * torch.sqrt(top_vals).unsqueeze(0)
                self.Lambda.data[k] = L_k_updated
                
                recon_cov = L_k_updated @ L_k_updated.T
                psi_update = torch.diagonal(Sigma_k) - torch.diagonal(recon_cov)
                psi_update = torch.clamp(psi_update, min=1e-3)
                self.log_psi.data[k] = torch.log(psi_update)
                
            except Exception as e:
                pass

        # Update Pi globally based on new S0 proportions
        self.log_pi.data = torch.log(self.S0 / self.S0.sum() + 1e-10)
    
    def compute_component_log_likelihoods(self, X):
        """
        Calculates the pure structural/geometric fit using the Woodbury Matrix Identity
        to avoid O(D^3) inversion of the 120x120 covariance matrix.
        This calculates log P(x_i | w_j) and entirely ignores the global mixing weights (pi).
        """
        N, D = X.shape
        log_probs = []
        
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
            
            # 4. Final Log PDF
            log_prob = -0.5 * (c + log_det_C + mahalanobis)
            log_probs.append(log_prob)
            
        return torch.stack(log_probs, dim=1)
    
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
        
    def add_component(self, new_mu, new_Lambda, new_log_psi, new_S0, new_S1, new_S2):
        """
        Dynamically adds a new component to the Mixture Model and integrates its sufficient statistics.
        """
        with torch.no_grad():
            q_new = new_Lambda.shape[2]
            
            # 1. Handle Tensor Shape Matching (Padding Lambda)
            if q_new < self.q:
                pad_size = self.q - q_new
                new_Lambda = torch.nn.functional.pad(new_Lambda, (0, pad_size))
            elif q_new > self.q:
                pad_size = q_new - self.q
                self.Lambda = nn.Parameter(torch.nn.functional.pad(self.Lambda.data, (0, pad_size)))
                self.q = q_new

            # === TRACK LOCAL DIMENSIONALITY ===
            # Append the new component's required factors to the tracking buffer
            new_q_tensor = torch.tensor([q_new], dtype=torch.long, device=self.device)
            self.q_k = torch.cat([self.q_k, new_q_tensor])

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
            log_resp_norm, _, _ = self.e_step(X)
            resp = torch.exp(log_resp_norm) 
            N = X.shape[0]
            
            # ALL statistics must be Expectations (batch averages)
            self.S0 = resp.mean(dim=0)          # Shape: (K,) - Sums to 1
            self.S1 = (resp.T @ X) / N          # Shape: (K, D)
            
            for k in range(self.K):
                resp_k = resp[:, k:k+1] 
                self.S2[k] = (resp_k * X).T @ X / N # Shape: (D, D)
            
            self.update_counts = torch.ones(self.K, device=self.device)
            
        print(f"Sufficient statistics initialized. Model total mass (S0 sum): {self.S0.sum().item():.2f}")
    
    
    
    
