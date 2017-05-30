import numpy as np

class posterior_stan(object):
    def __init__(self,fit,priors):
        """ Class for dealing with posterior from pystan

        :param fit: fit object from pystan
        :param priors: list of prior classes used for fit
        """
        self.nsrc=priors[0].nsrc
        self.samples=fit.extract()
        self.convergence_stats(fit)
        self.param_names=fit.model_pars
        self.scale_posterior(priors)
        self.ID=priors[0].ID
    
    def convergence_stats(self,fit):
        """Extract convergence statistics from the pystan fit object

        :param fit: fit object from pystan
        """
        converge=np.array(fit.summary(probs=[0.025, 0.15, 0.25, 0.5, 0.75, 0.84, 0.975])['summary'][:,:])
        self.converge=converge
        self.Rhat=converge[:,-1]
        self.n_eff=converge[:,-2]

    
    # define a function to get percentile for a particular parameter
    def quantileGet(self,q):

        """get percentile (q) for all fitted parameters

        :param q: percentile e.g. 16,50,84..
        :return: array containing percentile for parameter
        """
        samples,nparam=self.stan_fit.shape
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
        #create indices for posterior (i.e. include backgrounds and sigma_conf)
        """Stan searches over range 0-1 and scales parameters with flux limits. This function scales those parameters to flux values

        :param priors: list of prior classes used in fit

        """

        for i in range(0,len(priors)):
            lower=priors[i].prior_flux_lower
            upper=priors[i].prior_flux_upper
            self.samples['src_f'][:,i,:]=lower+(upper-lower)*self.samples['src_f'][:,i,:]
