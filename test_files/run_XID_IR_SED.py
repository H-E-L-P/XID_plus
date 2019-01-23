from astropy.io import ascii, fits
import pylab as plt
%matplotlib inline
from astropy import wcs


import numpy as np
import xidplus
from xidplus import moc_routines
import pickle

from xidplus import sed
SEDs, df=sed.berta_templates()
priors,posterior=xidplus.load(filename='./XID+SED_prior.pkl')
import xidplus.stan_fit.SED as SPM
fit=SPM.MIPS_PACS_SPIRE(priors,SEDs,chains=4,iter=1000)
posterior=sed.posterior_sed(fit,priors,SEDs)
xidplus.save(priors, posterior, 'test_SPM')