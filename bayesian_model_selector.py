import torch
import torch.nn as nn
from mfa import MFA  # Import your standard MFA class

class BayesianMFA_Initializer(MFA): # Inherit from MFA!
    def __init__(self, n_components, n_channels, q_max, tol=1e-4, max_iter=150, device='cpu'):
        # Call the parent MFA constructor to set up mu, Lambda, log_psi, log_pi, and e_step!
        super().__init__(
            n_components=n_components, 
            n_channels=n_channels, 
            n_factors=q_max, 
            tol=tol, 
            max_iter=max_iter, 
            device=device
        )
        
        # ARD Precision Hyperparameter (Alpha) - One for each latent factor, per component
        # Initialized to a small precision (high variance)
        self.alpha = torch.ones(self.K, self.q, device=self.device) * 1e-3 

    def fit_with_ard(self, X):
        """
        Batch EM with Automatic Relevance Determination to find optimal K and q.
        """
        X = X.to(self.device)
        prev_ll = -float('inf')
        
        for i in range(self.max_iter):
            # 1. Standard E-Step (Inherited directly from your mfa.py)
            log_resp_norm, log_likelihood, _, _ = self.e_step(X)
            resp = torch.exp(log_resp_norm) 
            
            # 2. Custom M-Step with ARD Penalty
            self._m_step_ard(X, resp)
            
            # 3. Update ARD Precisions (Alpha)
            self._update_ard_precisions()
            
            # 4. Check for convergence
            current_ll = log_likelihood.mean().item()
            if i > 0 and abs(current_ll - prev_ll) < self.tol:
                print(f"Bayesian Initialization converged at iteration {i}.")
                break
            prev_ll = current_ll

    def _m_step_ard(self, X, resp):
        """
        Custom M-step that incorporates the ARD prior penalty (alpha) into the Lambda update.
        """
        N = X.shape[0]
        Nk = resp.sum(dim=0) + 1e-10 # Prevent division by zero
        
        # Update mixing weights (pi)
        self.log_pi.data = torch.log(Nk / N)
        
        for k in range(self.K):
            resp_k = resp[:, k].unsqueeze(1) # (N, 1)
            
            # Update Mean (mu)
            mu_k = (X * resp_k).sum(dim=0) / Nk[k] # (D,)
            self.mu.data[k] = mu_k
            
            X_centered = X - mu_k # (N, D)
            
            # Current component parameters
            psi_k = torch.exp(self.log_psi[k]) # (D,)
            Lambda_k = self.Lambda[k] # (D, q)
            
            # Woodbury identity for inverse covariance
            psi_inv = 1.0 / (psi_k + 1e-10)
            L_T_psi_inv = Lambda_k.T * psi_inv.unsqueeze(0) # (q, D)
            
            M = torch.eye(self.q, device=self.device) + L_T_psi_inv @ Lambda_k # (q, q)
            M_inv = torch.linalg.inv(M) # (q, q)
            
            # Expected latent variables
            beta = M_inv @ L_T_psi_inv # (q, D)
            E_Z = X_centered @ beta.T # (N, q)
            
            # E[ZZ^T]
            E_ZZT = Nk[k] * M_inv + (E_Z.T * resp_k.T) @ E_Z # (q, q)
            
            # ==========================================
            # ARD UPDATE FOR LAMBDA
            # Standard FA: Lambda = (X_centered^T * E[Z]) @ E[ZZ^T]^-1
            # ARD FA:      Lambda = (X_centered^T * E[Z]) @ (E[ZZ^T] + diag(alpha_k))^-1
            # ==========================================
            X_T_EZ = (X_centered * resp_k).T @ E_Z # (D, q)
            
            # Add the ARD penalty (alpha) to the diagonal of E[ZZ^T]
            ARD_penalty = torch.diag(self.alpha[k]) 
            
            # This is where the magic happens: high alpha forces columns of Lambda to 0
            Lambda_new = X_T_EZ @ torch.linalg.inv(E_ZZT + ARD_penalty)
            self.Lambda.data[k] = Lambda_new
            
            # Update Noise Variance (Psi)
            S_k = ((X_centered ** 2) * resp_k).sum(dim=0) / Nk[k] # (D,)
            L_EZ_X = (Lambda_new * (E_Z.T @ (X_centered * resp_k)).T / Nk[k]).sum(dim=1)
            
            psi_new = S_k - L_EZ_X
            psi_new = torch.clamp(psi_new, min=1e-6) # Prevent negative variances
            self.log_psi.data[k] = torch.log(psi_new)

    def _update_ard_precisions(self):
        """
        Updates the precision alpha based on the current factor loadings.
        If a column in Lambda shrinks towards zero, alpha becomes huge, driving it identically to 0 next step.
        """
        with torch.no_grad():
            for k in range(self.K):
                # alpha_k = D / sum(Lambda_k^2 over D)
                lambda_sq_sum = torch.sum(self.Lambda[k] ** 2, dim=0) + 1e-10
                self.alpha[k] = self.D / lambda_sq_sum