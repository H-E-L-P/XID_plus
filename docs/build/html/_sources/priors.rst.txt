Beyond Positional Priors
========================

XID+ has been designed such that additional prior information can be added (hence the +).
This additional prior information could be on observables such as flux, or additional prior information on the model such as expected SEDs, redshifts etc.

XID+CIGALE
----------
In collaboration with Will Pearson, (`Pearson et al 2017 <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1704.02192>`_) we have looked into introducing independent (i.e. no covariance between bands) flux priors for the SPIRE bands, by fitting ancillary optical and near-IR data with the SED fitting code, CIGALE.

XID+SED
---------------------
There is also an SED extension to XID+, which uses an emulated version of CIGALE to fit physical parameters such as starformation rate, redshift and AGN fraction directly on the map
 .. toctree::
   :maxdepth: 4

   ./notebooks/examples/XID+SED_example.ipynb

XID+Extended Sources
--------------------
An extension that can handle extended sources is actively being developed by `Raphael Shirley <http://www.sussex.ac.uk/profiles/405997>`_
