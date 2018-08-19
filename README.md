# Gaussian Processes for Orientation Preference Maps

This package is a Python implementation of a Gaussian Process (GP) method for inferring cortical maps ([Macke et al., 2011](https://www.sciencedirect.com/science/article/pii/S1053811910007007)), based on their original [MATLAB-code](https://bitbucket.org/mackelab/gp_maps/). Compared to conventional (vector averaging) approaches, the method computes better maps from little data and can be used to quantify the uncertainty in an estimated orientation preference map (OPM). It also provides a principled way to choose the parameters in the smoothing step and incorporate inhomogeneous noise variances as well as noise correlations.

The basic idea is to specify a prior over OPMs as a GP. This means that a-priori assumptions about correlations of the pixels in a map are specified as a covariance function K. In this case, we use a Difference of Gaussians (DoG) kernel. This covariance function encodes our assumptions that OPMs are smooth and have a semi-periodic structure. This prior is then updated via Bayes' rule with imaging data to obtain a smooth posterior estimate of an OPM. 

My notes on the method (including derivations not in the original paper) can be found [here](https://gitlab.com/dominikstrb/orientation_preference_maps/blob/master/SupportingInfo.pdf).

A number of notebooks illustrate how to use the package:

## [OPM.ipynb](https://gitlab.com/dominikstrb/orientation_preference_maps/blob/master/OPM.ipynb)

contains basic functionality for handling OPMs:

- sample from this prior to obtain a ground truth OPM
- visualize OPMs
- simulate neural responses given an OPM and a set of stimuli
- compute maximum likelihood/vector averaging estimate of the OPM
- smooth the OPM with a Gaussian kernel


## [BayesianOPM.ipynb](https://gitlab.com/dominikstrb/orientation_preference_maps/blob/master/BayesianOPM.ipynb)

shows details about the GP method:

- covariance function (DoG kernel)
- estimate hyperparameters of the prior covariance function from the empirical map
- learn noise model iteratively from data
- posterior inference

## [FerretData.ipynb](https://gitlab.com/dominikstrb/orientation_preference_maps/blob/master/FerretData.ipynb)

serves as an illustration on how to use GPs to infer OPMs from real imaging data (replicating the results of Macke et al., 2011):

- visualize inhomogeneous and correlated noise structure of the data
- obtain smooth estimate based on "ground truth"
- GP posterior inference on real data





