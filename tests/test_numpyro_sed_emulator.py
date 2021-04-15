import unittest
import xidplus
from xidplus.numpyro_fit import SPIRE, SED_prior
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
import numpy as np
import numpyro

class test_all_bands(unittest.TestCase):
    def setUp(self):

        priors, posterior = xidplus.load('test.pkl')
        self.priors = priors
        self.posterior=posterior

    def test_all_bands(self):
        fit=SED_prior.all_bands(self.priors,'./GB_emulator_20210106.pkl')
        fit.print_summary()
        posterior_numpyro=xidplus.posterior_numpyro_sed(fit,self.priors,'./GB_emulator_20210106.pkl')


if __name__ == '__main__':
    unittest.main()
