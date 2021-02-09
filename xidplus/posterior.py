import numpy as np


class posterior_stan(object):
    def __init__(self,fit,priors, scale=True):
        """ Class for dealing with posterior from pystan

        :param fit: fit object from pystan
        :param priors: list of prior classes used for fit
        """
        self.nsrc=priors[0].nsrc
        self.samples=fit.extract()
        if len(priors) < 2:
            self.samples['bkg']=self.samples['bkg'][:,None]
            self.samples['sigma_conf'] = self.samples['sigma_conf'][:, None]
        self.param_names=fit.model_pars

        self.ID=priors[0].ID
        self.Rhat = {'src_f': fit.summary('src_f')['summary'][:, -1].reshape(priors[0].nsrc, len(priors)),
                     'sigma_conf': fit.summary('sigma_conf')['summary'][:, -1],
                     'bkg': fit.summary('bkg')['summary'][:, -1]}

        self.n_eff = {'src_f': fit.summary('src_f')['summary'][:, -2].reshape(priors[0].nsrc, len(priors)),
                      'sigma_conf': fit.summary('sigma_conf')['summary'][:, -2],
                      'bkg': fit.summary('bkg')['summary'][:, -2]}
        if scale is True:
            self.scale_posterior(priors)

    def scale_posterior(self,priors):
        #create indices for posterior (i.e. include backgrounds and sigma_conf)
        """Stan searches over range 0-1 and scales parameters with flux limits. This function scales those parameters to flux values

        :param priors: list of prior classes used in fit

        """

        for i in range(0,len(priors)):
            lower=priors[i].prior_flux_lower
            upper=priors[i].prior_flux_upper
            self.samples['src_f'][:,i,:]=lower+(upper-lower)*self.samples['src_f'][:,i,:]


class posterior_pyro(object):
    def __init__(self, fit, priors):
        """ Class for dealing with posterior from pyro

        :param fit: fit object from pyrop
        :param priors: list of prior classes used for fit
        """
        self.nsrc=priors[0].nsrc
        self.samples=fit['samples']
        self.loss_history=fit['loss_history']
        if len(priors) < 2:
            self.samples['bkg']=self.samples['bkg'][:,None]
            self.samples['sigma_conf'] = self.samples['sigma_conf'][:, None]


class posterior_numpyro(object):
    def __init__(self, mcmc, priors):
        from numpyro.diagnostics import print_summary, summary

        from operator import attrgetter

        """ Class for dealing with posterior from numpyro

        :param fit: fit object from numpyro
        :param priors: list of prior classes used for fit
        """
        self.nsrc=priors[0].nsrc
        self.samples=mcmc.get_samples()
        self.samples['src_f']=np.swapaxes(self.samples['src_f'],1,2)
        # get summary statistics. Code based on numpyro print_summary
        prob = 0.9
        exclude_deterministic = True
        sites = mcmc._states[mcmc._sample_field]
        if isinstance(sites, dict) and exclude_deterministic:
            state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
            # XXX: there might be the case that state.z is not a dictionary but
            # its postprocessed value `sites` is a dictionary.
            # TODO: in general, when both `sites` and `state.z` are dictionaries,
            # they can have different key names, not necessary due to deterministic
            # behavior. We might revise this logic if needed in the future.
            if isinstance(state_sample_field, dict):
                sites = {k: v for k, v in mcmc._states[mcmc._sample_field].items()
                         if k in state_sample_field}

        stats_summary = summary(sites, prob=prob)
        diverge = mcmc.get_extra_fields()['diverging']

        self.Rhat = {'src_f': stats_summary['src_f']['r_hat'],
                     'sigma_conf': stats_summary['sigma_conf']['r_hat'],
                     'bkg': stats_summary['bkg']['r_hat']}

        self.n_eff = {'src_f': stats_summary['src_f']['n_eff'],
                     'sigma_conf': stats_summary['sigma_conf']['n_eff'],
                     'bkg': stats_summary['bkg']['n_eff']}
        self.divergences=diverge
        print("Number of divergences: {}".format(np.sum(diverge)))

        if len(priors) < 2:
            self.samples['bkg']=self.samples['bkg'][:,None]
            self.samples['sigma_conf'] = self.samples['sigma_conf'][:, None]

class posterior_numpyro_sed(object):
    def __init__(self, mcmc, priors, sed_prior):
        from numpyro.diagnostics import summary
        import jax.numpy as jnp

        from operator import attrgetter

        """ Class for dealing with posterior from numpyro

        :param fit: fit object from numpyro
        :param priors: list of prior classes used for fit
        """
        self.nsrc=priors[0].nsrc
        self.samples=mcmc.get_samples()
        self.samples['src_f']=jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'],sed_prior.params_mu+self.samples['params']*sed_prior.params_sig))
        self.samples['src_f']=np.swapaxes(self.samples['src_f'],1,2)
        # get summary statistics. Code based on numpyro print_summary
        prob = 0.9
        exclude_deterministic = True
        sites = mcmc._states[mcmc._sample_field]
        if isinstance(sites, dict) and exclude_deterministic:
            state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
            # XXX: there might be the case that state.z is not a dictionary but
            # its postprocessed value `sites` is a dictionary.
            # TODO: in general, when both `sites` and `state.z` are dictionaries,
            # they can have different key names, not necessary due to deterministic
            # behavior. We might revise this logic if needed in the future.
            if isinstance(state_sample_field, dict):
                sites = {k: v for k, v in mcmc._states[mcmc._sample_field].items()
                         if k in state_sample_field}

        stats_summary = summary(sites, prob=prob)
        diverge = mcmc.get_extra_fields()['diverging']

        self.Rhat = {'params': stats_summary['params']['r_hat'],
                     'sigma_conf': stats_summary['sigma_conf']['r_hat'],
                     'bkg': stats_summary['bkg']['r_hat']}

        self.n_eff = {'params': stats_summary['params']['n_eff'],
                     'sigma_conf': stats_summary['sigma_conf']['n_eff'],
                     'bkg': stats_summary['bkg']['n_eff']}
        self.divergences=diverge
        print("Number of divergences: {}".format(np.sum(diverge)))

        if len(priors) < 2:
            self.samples['bkg']=self.samples['bkg'][:,None]
            self.samples['sigma_conf'] = self.samples['sigma_conf'][:, None]


