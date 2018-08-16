# Gaussian Processes for Orientation Preference Maps

This package is a Python implementation of a Gaussian Process (GP) method for inferring cortical maps ([Macke et al., 2011](https://www.sciencedirect.com/science/article/pii/S1053811910007007)), based on their original [MATLAB-code](https://bitbucket.org/mackelab/gp_maps/). Compared to conventional (vector averaging) approaches, the method computes better maps from little data and can be used to quantify the uncertainty in an estimated orientation preference map (OPM). It also provides a principled way to choose the parameters in the smoothing step and incorporate inhomogeneous noise variances as well as noise correlations.

The basic idea is to specify a prior over OPMs as a GP. This means that a-priori assumptions about correlations of the pixels in a map are specified as a covariance function K. In this case, we use a Difference of Gaussians (DoG) kernel. This covariance function encodes our assumptions that OPMs are smooth and have a semi-periodic structure. For simplicity, the imaginary and real map component are assumed to be a-priori independent. We can generate samples from this prior

```python
from opm import make_opm, plot_opm

m = make_opm(size=(100,100), sigma=8., k=2., alpha=1.)
plot_opm(m)
```



