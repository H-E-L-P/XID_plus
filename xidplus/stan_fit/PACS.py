__author__ = 'pdh21'
import os
import numpy as np
from numpy import long

output_dir=os.getcwd()

full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def all_bands(PACS_100,PACS_160,chains=4,iter=1000):
    """
    Fit the two PACS bands

    :param PACS_100: xidplus.prior class
    :param PACS_160: xidplus.prior class
    :param chains: number of chains
    :param iter: number of samples
    :return: pystan fit object
    """

    #input data into a dictionary

    #XID_data={'nsrc':PACS_100.nsrc,
    #      'npix_psw':PACS_100.snpix,
    #      'nnz_psw':PACS_100.amat_data.size,
    #      'db_psw':PACS_100.sim,
    #      'sigma_psw':PACS_100.snim,
    #      'bkg_prior_psw':PACS_100.bkg[0],
    #      'bkg_prior_sig_psw':PACS_100.bkg[1],
    #      'Val_psw':PACS_100.amat_data,
    #      'Row_psw': PACS_100.amat_row.astype(np.long),
    #      'Col_psw': PACS_100.amat_col.astype(np.long),
    #      'f_low_lim_psw': PACS_100.prior_flux_lower,
    #      'f_up_lim_psw': PACS_100.prior_flux_upper,
    #      'npix_pmw':PACS_160.snpix,
    #      'nnz_pmw':PACS_160.amat_data.size,
    #      'db_pmw':PACS_160.sim,
    #      'sigma_pmw':PACS_160.snim,
    #      'bkg_prior_pmw':PACS_160.bkg[0],
    #      'bkg_prior_sig_pmw':PACS_160.bkg[1],
    #      'Val_pmw':PACS_160.amat_data,
    #      'Row_pmw': PACS_160.amat_row.astype(np.long),
    #      'Col_pmw': PACS_160.amat_col.astype(np.long),
    #      'f_low_lim_pmw': PACS_160.prior_flux_lower,
    #      'f_up_lim_pmw': PACS_160.prior_flux_upper}

    XID_data = {'nsrc': PACS_100.nsrc,
                'f_low_lim': [PACS_100.prior_flux_lower, PACS_160.prior_flux_lower],
                'f_up_lim': [PACS_100.prior_flux_upper, PACS_160.prior_flux_upper],
                'bkg_prior': [PACS_100.bkg[0], PACS_160.bkg[0]],
                'bkg_prior_sig': [PACS_100.bkg[1], PACS_160.bkg[1]],
                'npix_psw': PACS_100.snpix,
                'nnz_psw': PACS_100.amat_data.size,
                'db_psw': PACS_100.sim,
                'sigma_psw': PACS_100.snim,
                'Val_psw': PACS_100.amat_data,
                'Row_psw': PACS_100.amat_row.astype(long),
                'Col_psw': PACS_100.amat_col.astype(long),
                'npix_pmw': PACS_160.snpix,
                'nnz_pmw': PACS_160.amat_data.size,
                'db_pmw': PACS_160.sim,
                'sigma_pmw': PACS_160.snim,
                'Val_pmw': PACS_160.amat_data,
                'Row_pmw': PACS_160.amat_row.astype(long),
                'Col_pmw': PACS_160.amat_col.astype(long)}
    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+PACS'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)

    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True,init='random')
    #return fit data
    return fit

def all_bands_gaussian(PACS_100,PACS_160,chains=4,iter=1000,optimise=False):
    """
    Fit the two PACS bands with informed prior

    :param PACS_100: xidplus.prior class
    :param PACS_160: xidplus.prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    import numpy as np
    
    f_mu_100 = (PACS_100.prior_flux_mu - PACS_100.prior_flux_lower) / (PACS_100.prior_flux_upper - PACS_100.prior_flux_lower)
    f_mu_160 = (PACS_160.prior_flux_mu - PACS_160.prior_flux_lower) / (PACS_160.prior_flux_upper - PACS_160.prior_flux_lower)
    
    f_sigma_100 = PACS_100.prior_flux_sigma / (PACS_100.prior_flux_upper - PACS_100.prior_flux_lower)
    f_sigma_160 = PACS_160.prior_flux_sigma / (PACS_160.prior_flux_upper - PACS_160.prior_flux_lower)

    XID_data={'nsrc':PACS_100.nsrc,
              'f_low_lim':[PACS_100.prior_flux_lower,PACS_160.prior_flux_lower],
              'f_up_lim':[PACS_100.prior_flux_upper,PACS_160.prior_flux_upper],
              'f_mu':[f_mu_100,f_mu_160],
              'f_sigma':[f_sigma_100,f_sigma_160],
              'bkg_prior':[PACS_100.bkg[0],PACS_160.bkg[0]],
              'bkg_prior_sig':[PACS_100.bkg[1],PACS_160.bkg[1]],
          'npix_psw':PACS_100.snpix,
          'nnz_psw':PACS_100.amat_data.size,
          'db_psw':PACS_100.sim,
          'sigma_psw':PACS_100.snim,
          'Val_psw':PACS_100.amat_data,
          'Row_psw': PACS_100.amat_row.astype(long),
          'Col_psw': PACS_100.amat_col.astype(long),
          'npix_pmw':PACS_160.snpix,
          'nnz_pmw':PACS_160.amat_data.size,
          'db_pmw':PACS_160.sim,
          'sigma_pmw':PACS_160.snim,
          'Val_pmw':PACS_160.amat_data,
          'Row_pmw': PACS_160.amat_row.astype(long),
          'Col_pmw': PACS_160.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+PACS_gaussian'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)
    
    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)

    #return fit data
    return fit
