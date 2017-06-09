Beyond Positional Priors
========================

XID+ has been designed such that additional prior information can be added (hence the +).
This additional prior information could be on observables such as flux, or additional prior information on the model such as expected SEDs, redshifts etc.

XID+CIGALE
----------
In collaboration with Will Pearson, (`Pearson et al 2017 <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1704.02192>`_) we have looked into introducing independent (i.e. no covariance between bands) flux priors for the SPIRE bands, by fitting ancillary optical and near-IR data with the SED fitting code, CIGALE.

XID+SPIRE Prior Model
---------------------
There is also an SED based model extension to XID+, which takes in SEDs and a prior in redshift. This model has the added benefit of
providing a physical correlation between bands and appropriate physical limits based on redshift and infrared luminosity.
As an additional output it also provides a posterior probability density function for redshift and infrared luminosity.

If you interested in using XID+ with additional priors, please contact `Peter Hurley <http://www.sussex.ac.uk/profiles/188689>`_

XID+Extended Sources
--------------------
An extension that can handle extended sources is actively being developed by `Raphael Shirley <http://www.sussex.ac.uk/profiles/405997>`_
