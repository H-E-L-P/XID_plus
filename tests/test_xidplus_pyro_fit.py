import unittest
import xidplus
from xidplus.pyro_fit import SPIRE


class test_all_bands(unittest.TestCase):
    def setUp(self):
        priors, posterior = xidplus.load('test.pkl')
        self.priors=priors

    def test_spire_model(self):

        psw,pmw,plw=SPIRE.spire_model(self.priors)
        self.assertEqual(psw.size()[0], self.priors[0].snpix)
        self.assertEqual(pmw.size()[0],self.priors[1].snpix)
        self.assertEqual(plw.size()[0], self.priors[2].snpix)

    def test_spire_model_wrong_prior_length(self):
        self.assertRaises(ValueError,lambda: SPIRE.spire_model(self.priors[0:2]))

    def test_all_bands(self):
        fit=SPIRE.all_bands(self.priors,n_steps=100)
        self.assertEqual(len(fit['loss_history']),100)
        self.assertIsNotNone(xidplus.posterior_pyro(fit,self.priors))






if __name__ == '__main__':
    unittest.main()
