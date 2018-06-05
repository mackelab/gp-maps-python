from scipy.signal import correlate2d
from scipy.optimize import curve_fit
import numpy as np
from kernels import mexican_hat_kernel

def rot_avg(xcorr, spacing=None):
    
    
    s1 = (xcorr.shape[0] - 1) / 2
    s2 = (xcorr.shape[1] - 1) / 2
    x, y = np.meshgrid(np.arange(-s2, s2+1), np.arange(-s1, s1+1))

    rs = np.sqrt(x**2 + y**2)
    
    if spacing is None:
        
        r = np.unique(rs)
        n = np.zeros(r.size)
        rotmean = np.zeros(r.size)
        rotstd = np.zeros(r.size)
        
        for k in range(r.size):

            values = xcorr[rs == r[k]]
            rotmean[k] = values.mean()
            n[k] = values.size
            rotstd[k] = values.std()
            
    else:   
        
        r = np.zeros(spacing.size - 1)
        n = np.zeros(spacing.size - 1)
        rotmean = np.zeros(spacing.size - 1)
        rotstd = np.zeros(spacing.size - 1)
        
        for k in range(spacing.size - 1):
            r[k] = spacing[k]
            
            values = xcorr[(rs >= spacing[k] - 1e-30) & (rs < spacing[k+1])]
                           
            rotmean[k] = values.mean()
            n[k] = values.size
            rotstd[k] = values.std()
    
    return rotmean, r, rotstd, n



def radial_component(a, spacing=None):
    if spacing is None:
        maxr = int(np.floor(np.sqrt(a.shape[0]**2 + a.shape[1]**2)))
        spacing = np.arange(maxr)
        
    mean, r, std, n = rot_avg(correlate2d(a, a) / a.size, spacing=spacing)
    return mean, r


def match_radial_component(m, kernel=mexican_hat_kernel, p0=None):
    
    # if none are given, use default arguments for mexican hat kernel with reasonable initial values
    if p0 is None:
        p0 = {'sigma': 3., 'alpha': 2.}

    # compute maximal distance in the map
    maxr = int(np.floor(np.sqrt(m.shape[0]**2 + m.shape[1]**2)))
    
    rot_cov = np.zeros((2,maxr-1))
    
    # for each map component, compute the rotational average of the autocorrelation
    for i, m_i in enumerate([np.real(m), np.imag(m)]):
        xcorr = correlate2d(m_i, m_i) / m_i.size
        mean, r = radial_component(m_i, spacing=np.arange(maxr))
        rot_cov[i] = mean
        
    # mean across both map components
    rot_cov = rot_cov.mean(axis=0)
    
    def f(r, *params):
        # get argument names from given dict and values passed to this function
        kernel_kwargs = {name: value for name, value in zip(p0.keys(), params)}
        return kernel(r[:,np.newaxis], 0, **kernel_kwargs)

    # fit the kernel to the empirical map
    popt, pcov = curve_fit(f, xdata=r, ydata=rot_cov, p0=list(p0.values()))
    sigma_opt, alpha_opt = popt
    
    return sigma_opt, alpha_opt