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

    def test_spire_model(self):
        nuts_kernel = NUTS(SPIRE.spire_model)
        mcmc = MCMC(nuts_kernel,num_samples=100,num_warmup=100)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key,self.priors )
        posterior_samples = mcmc.get_samples()
        self.assertIsNotNone(posterior_samples)
        self.assertEqual(posterior_samples['src_f'].shape[1], self.priors[0].nsrc)
    def test_all_bands(self):
        fit=SPIRE.all_bands(self.priors)
        fit.print_summary()
        posterior_numpyro=xidplus.posterior_numpyro(fit,self.priors)
        self.assertAlmostEqual(np.mean(posterior_numpyro.samples['bkg']),np.mean(self.posterior.samples['bkg']),places=0)

    def test_spire_model_prior_pred_sample(self):
        for i in range(0,3):
            self.priors[i].sim=None
        fit=SPIRE.all_bands(self.priors)
        fit.print_summary()

if __name__ == '__main__':
    unittest.main()
