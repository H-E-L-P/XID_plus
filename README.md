
# XID+, The Probabilistic De-blender for confusion dominated maps
=================================================================

Read the documentation at: [herschel.sussex.ac.uk/XID_plus](http://herschel.sussex.ac.uk/XID_plus)

## Contributors

This code is being developed by [Peter Hurley](http://www.sussex.ac.uk/profiles/188689). 

## [License](Licence.md)

## Gaussian Priors

Outlined in [Pearson et al. 2017](http://adsabs.harvard.edu/abs/2017A%26A...603A.102P)  
Adds `flux_mu` and `flux_sigma` parameters into the prior to hold the mean (mu) and error (sigma) of the flux prior.  
To use the gaussain prior, add `_gaussian` to the call to the fitting routine.  
  e.g. `SPIRE.all_bands_gaussian(prior250, prior350, prior500)` instead of `SPIRE.all_bands(prior250, prior350, prior500)`
