import numpy as np

class posterior_stan(object):
    def __init__(self,fit,priors):
        """ Class for dealing with posterior from stan"""
        self.nsrc=priors[0].nsrc
        self.convergence_stats(fit)
        self.param_names=fit.constrained_param_names()
        self.stan_fit=fit.extract(permuted=False, inc_warmup=False)
        self.scale_posterior(priors)
    
    def convergence_stats(self,fit):
        converge=np.array(fit.summary(probs=[0.025, 0.15, 0.25, 0.5, 0.75, 0.84, 0.975])['summary'][:,:])
        self.Rhat=converge[:,-1]
        self.n_eff=converge[:,-2]

    
    # define a function to get percentile for a particular parameter
    def quantileGet(self,q):
        chains,iter,nparam=self.stan_fit.shape
        param=self.stan_fit.reshape((chains*iter,nparam))
        #q is quantile
        #param is array (nsamples,nparameters)
        # make a list to store the quantiles
        quants = []
 
        # for every predicted value
        for i in range(param.shape[1]):
            # make a vector to store the predictions from each chain
            val = []
 
            # next go down the rows and store the values
            for j in range(param.shape[0]):
                val.append(param[j,i])
 
            # return the quantile for the predictions.
            quants.append(np.percentile(val, q))
 
        return quants
    
    def covariance_sparse(self,threshold=0.1):
        """Create sparse covariance matrix from posterior. \n 
        Only stores values that are greater than given threshold (default=|0.1|)"""
        chains,iter,nparam=self.stan_fit.shape
        #Create index for sources that correspond to index in covariance matrix
        ij=np.append(np.arange(0,self.nsrc+1),[np.arange(0,self.nsrc+1),np.arange(0,self.nsrc+1)])
        #Create index for band that correspond to index in covarariance matrix
        bb=np.append(np.full(self.nsrc+1,0),[np.full(self.nsrc+1,1),np.full(self.nsrc+1,2)])
        i_cov,j_cov=np.meshgrid(ij,ij)
        k_cov,l_cov=np.meshgrid(bb,bb)
        #Calculate covariance matrix
        cov=np.cov(self.stan_fit.reshape((chains*iter,nparam)).T)
        #Rather than storing full cov matrix, use only upper triangle (and diag)
        cov=np.triu(cov,0) #this sets lower tri to zero
        #select elements greater than threshold
        index=np.abs(cov)>threshold
        self.XID_i=i_cov[index]
        self.XID_j=j_cov[index]
        self.Band_k=k_cov[index]
        self.Band_l=l_cov[index]
        self.sigma_i_j_k_l=cov[index]

    def scale_posterior(self,priors):
        #create indices for posterior (i.e. inlcude backgrounds and sigma_conf)
        ind=[True]*self.nsrc
        ind_tmp=np.array(ind+[False]+ind+[False]+ind+[False]+[False,False,False,False])

        lower=np.array([])
        upper=np.array([])

        #scale from 0-1 to flux values:
        for i in priors:
            lower=np.append(lower,i.prior_flux_lower)
            upper=np.append(upper,i.prior_flux_upper)

        # lower=np.append(np.append(tmp_prior250.prior_flux_lower,tmp_prior350.prior_flux_lower),tmp_prior500.prior_flux_lower)
        # upper=np.append(np.append(tmp_prior250.prior_flux_upper,tmp_prior350.prior_flux_upper),tmp_prior500.prior_flux_upper)

        if priors[0].scale == 'linear':
            self.stan_fit[:, :, ind_tmp] = lower + (upper - lower) * self.stan_fit[:, :, ind_tmp]
        else:
            self.stan_fit[:, :, ind_tmp] = np.power(10.0, lower + (upper - lower) * self.stan_fit[:, :, ind_tmp])



def scale_posterior(priors, posterior,log=True):
    #create indices for posterior (i.e. inlcude backgrounds and sigma_conf)
    ind=[True]*posterior.nsrc
    ind_tmp=np.array((ind+[False])*len(priors)+[False]*len(priors)+[False])

    lower=np.array([])
    upper=np.array([])

    #scale from 0-1 to flux values:
    for i in priors:
        lower=np.append(lower,i.prior_flux_lower)
        upper=np.append(upper,i.prior_flux_upper)

    # lower=np.append(np.append(tmp_prior250.prior_flux_lower,tmp_prior350.prior_flux_lower),tmp_prior500.prior_flux_lower)
    # upper=np.append(np.append(tmp_prior250.prior_flux_upper,tmp_prior350.prior_flux_upper),tmp_prior500.prior_flux_upper)

    if log is False:
        posterior.stan_fit[:, :, ind_tmp] = lower + (upper - lower) * posterior.stan_fit[:, :, ind_tmp]
    else:
        posterior.stan_fit[:, :, ind_tmp] = np.power(10.0, lower + (upper - lower) * posterior.stan_fit[:, :, ind_tmp])
    return posterior