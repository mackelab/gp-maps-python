from kernels import mexican_hat_kernel
from linalg import ICD


class LowRankPrior():
    
    def __init__(self, x, method=None, rank=None):
        """ Setup LowRank prior
        Args:
            method: currently allows 'icd' or None, defaults to None
            rank: rank of the approximation (not used when None is specified)
        """
        
        if method not in [None, 'icd']:
            raise ValueError('No valid low-rank method specified.')
        self.method = method
        self.x = x
        
        if not rank:
            self.rank = x.shape[0]
        else:
            self.rank = rank
            
        
            
    def fit(self, kernel=mexican_hat_kernel, ridge=1e-4, **kernel_kwargs):
        """ Learn a (low-rank) prior.
        Args:
            kernel: a kernel function that takes two vectors x and y
            ridge: element that gets added to the diagonal to avoid numerical instabilites (use depends on low-rank method)
            kernel_kwargs: get passed to kernel(...)
            
        After fitting, the prior has the following attributes:
            - G: Low rank approximation K = GG' of prior covariance matrix
            - K: Full prior covariance matrix (depending on low-rank method)
        """
    
        if not self.method:
            self.K = prior_covariance(self.x, kernel=kernel, **kernel_kwargs)
            self.G = ridge_cholesky(self.K)
            
        elif self.method.lower() == 'icd':
            icd = ICD(rank=self.rank)
            icd.fit(self.x, kernel, ridge, **kernel_kwargs)
            self.G = icd.G
            self.K = icd
            

    def __getitem__(self, idx):
        """ Access portions of K = GG'
        Args:
            item: indices to access K = GG'
        Return:
            K[idx[0], idx[i]]
        """

        return self.K[idx[0], idx[1]]
    
class GaussianProcessOPM():
    
    def __init__(self, size, prior_rank, prior_method='icd', kernel=mexican_hat_kernel, **kernel_kwargs):
        self.size = size
        self.idx = get_indices(size)
        
        self.rank = prior_rank
        
        self.kernel = kernel
        
        self.prior = LowRankPrior(self.idx, method=prior_method, rank=prior_rank)
        self.prior.fit(kernel=self.kernel, **kernel_kwargs)
        
    def fit(self, responses, noise_cov):
        R = responses
        
        G = self.prior.G
        K = G @ G.T
        beta = 2 / R.shape[0] 
        
        S = np.linalg.inv(noise_cov)
        
        K_post_single =  K - 1/beta * K @ (S - S @ G @ np.linalg.inv(beta * np.eye(self.rank) + G.T @ S @ G) @ G.T @ S) @ G @ G.T
        
        K_post = np.kron(np.eye(d), K_post_single)
        
        # inefficient version (commented out for readability)
        # K_post = np.linalg.inv(np.linalg.inv(K_m) + np.kron(N/2 * np.eye(d), K_e))
        
        vr = np.zeros((n*d,1))
        for v, r in zip(V, R):
            vr += np.kron(v, r)[:,np.newaxis]

        mu_post = K_post @ np.kron(np.eye(d), np.linalg.inv(K_e)) @ vr
        
        return mu_post
    
    

""" TODO: do we even need this?
# correct the diagonal
prior_var_uncorrected = np.zeros(n)
prior_var_exact = np.zeros(n)

for k in range(n):
    prior_var_uncorrected[k] = G[k,:] @ G[k,:].T
    prior_var_exact[k] = mexican_hat_kernel(idx[k], idx[k], sigma=2.)

diag = prior_var_exact - prior_var_uncorrected + 2 * 1e-4

K_m = np.kron(np.eye(d), G @ G.T + np.diag(diag))"""