import numpy as np

class posterior_stan(object):
    def __init__(self,fit,priors):
        """ Class for dealing with posterior from pystan

        :param fit: fit object from pystan
        :param priors: list of prior classes used for fit
        """
        self.nsrc=priors[0].nsrc
        self.samples=fit.extract()
        self.param_names=fit.model_pars
        self.scale_posterior(priors)
        
        self.ID=priors[0].ID
        self.Rhat = {'src_f': fit.summary('src_f')['summary'][:, -1].reshape(priors[0].nsrc, len(priors)),
                     'sigma_conf': fit.summary('sigma_conf')['summary'][:, -1],
                     'bkg': fit.summary('bkg')['summary'][:, -1]}

        self.n_eff = {'src_f': fit.summary('src_f')['summary'][:, -2].reshape(priors[0].nsrc, len(priors)),
                      'sigma_conf': fit.summary('sigma_conf')['summary'][:, -2],
                      'bkg': fit.summary('bkg')['summary'][:, -2]}
    

    def scale_posterior(self,priors):
        #create indices for posterior (i.e. include backgrounds and sigma_conf)
        """Stan searches over range 0-1 and scales parameters with flux limits. This function scales those parameters to flux values

        :param priors: list of prior classes used in fit

        """

        for i in range(0,len(priors)):
            lower=priors[i].prior_flux_lower
            upper=priors[i].prior_flux_upper
            self.samples['src_f'][:,i,:]=lower+(upper-lower)*self.samples['src_f'][:,i,:]
