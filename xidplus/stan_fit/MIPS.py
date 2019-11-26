from numpy import long

__author__ = 'pdh21'
import os
import numpy as np
output_dir=os.getcwd()


full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)


stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def MIPS_24(MIPS_24,chains=4,iter=1000):
    """
    Fit the MIPS 24 band

    :param MIPS_24: xidplus.prior class
    :param chains: number of chains
    :param iter:  number of iterations
    :return: pystan fit object
    """

    #input data into a dictionary
    XID_data={'nsrc':MIPS_24.nsrc,
              'f_low_lim':[MIPS_24.prior_flux_lower],
              'f_up_lim':[MIPS_24.prior_flux_upper],
              'bkg_prior':[MIPS_24.bkg[0]],
              'bkg_prior_sig':[MIPS_24.bkg[1]],
          'npix_psw':MIPS_24.snpix,
          'nnz_psw':MIPS_24.amat_data.size,
          'db_psw':MIPS_24.sim,
          'sigma_psw':MIPS_24.snim,
          'Val_psw':MIPS_24.amat_data,
          'Row_psw': MIPS_24.amat_row.astype(long),
          'Col_psw': MIPS_24.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it

    model_file= '/XID+MIPS'
    from xidplus.stan_fit import get_stancode
    sm= get_stancode(model_file)
    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit

def MIPS_24_gaussian(MIPS_24,chains=4,iter=1000,optimise=False):
    """
    Fit the MIPS24 band with informed prior

    :param MIPS_24: xidplus.prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    import numpy as np
    
    f_mu_24 = (MIPS_24.prior_flux_mu - MIPS_24.prior_flux_lower) / (MIPS_24.prior_flux_upper - MIPS_24.prior_flux_lower)
    
    f_sigma_24 = MIPS_24.prior_flux_sigma / (MIPS_24.prior_flux_upper - MIPS_24.prior_flux_lower)

    XID_data={'nsrc':MIPS_24.nsrc,
              'f_low_lim':[MIPS_24.prior_flux_lower],
              'f_up_lim':[MIPS_24.prior_flux_upper],
              'f_mu':[f_mu_24],
              'f_sigma':[f_sigma_24],
              'bkg_prior':[MIPS_24.bkg[0]],
              'bkg_prior_sig':[MIPS_24.bkg[1]],
          'npix_psw':MIPS_24.snpix,
          'nnz_psw':MIPS_24.amat_data.size,
          'db_psw':MIPS_24.sim,
          'sigma_psw':MIPS_24.snim,
          'Val_psw':MIPS_24.amat_data,
          'Row_psw': MIPS_24.amat_row.astype(long),
          'Col_psw': MIPS_24.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+MIPS_gaussian'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)
    
    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)

    #return fit data
    return fit
