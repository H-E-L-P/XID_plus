from numpy import long

__author__ = 'pdh21'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def all_bands(SPIRE_250,SPIRE_350,SPIRE_500,chains=4,iter=1000):

    """
    Fit the three SPIRE bands

    :param SPIRE_250: xidplus.prior class
    :param SPIRE_350: xidplus.prior class
    :param SPIRE_500: xidplus.prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """

    #input data into a dictionary

    XID_data={'nsrc':SPIRE_250.nsrc,
              'f_low_lim':[SPIRE_250.prior_flux_lower,SPIRE_350.prior_flux_lower,SPIRE_500.prior_flux_lower],
              'f_up_lim':[SPIRE_250.prior_flux_upper,SPIRE_350.prior_flux_upper,SPIRE_500.prior_flux_upper],
              'bkg_prior':[SPIRE_250.bkg[0],SPIRE_350.bkg[0],SPIRE_500.bkg[0]],
              'bkg_prior_sig':[SPIRE_250.bkg[1],SPIRE_350.bkg[1],SPIRE_500.bkg[1]],
          'npix_psw':SPIRE_250.snpix,
          'nnz_psw':SPIRE_250.amat_data.size,
          'db_psw':SPIRE_250.sim,
          'sigma_psw':SPIRE_250.snim,
          'Val_psw':SPIRE_250.amat_data,
          'Row_psw': SPIRE_250.amat_row.astype(long),
          'Col_psw': SPIRE_250.amat_col.astype(long),
          'npix_pmw':SPIRE_350.snpix,
          'nnz_pmw':SPIRE_350.amat_data.size,
          'db_pmw':SPIRE_350.sim,
          'sigma_pmw':SPIRE_350.snim,
          'Val_pmw':SPIRE_350.amat_data,
          'Row_pmw': SPIRE_350.amat_row.astype(long),
          'Col_pmw': SPIRE_350.amat_col.astype(long),
          'npix_plw':SPIRE_500.snpix,
          'nnz_plw':SPIRE_500.amat_data.size,
          'db_plw':SPIRE_500.sim,
          'sigma_plw':SPIRE_500.snim,
          'Val_plw':SPIRE_500.amat_data,
          'Row_plw': SPIRE_500.amat_row.astype(long),
          'Col_plw': SPIRE_500.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+SPIRE'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit

def all_bands_log10(SPIRE_250,SPIRE_350,SPIRE_500,chains=4,iter=1000):

    """
    Fit the three SPIRE bands

    :param SPIRE_250: xidplus.prior class
    :param SPIRE_350: xidplus.prior class
    :param SPIRE_500: xidplus.prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """

    #input data into a dictionary

    XID_data={'nsrc':SPIRE_250.nsrc,
              'f_low_lim':[SPIRE_250.prior_flux_lower,SPIRE_350.prior_flux_lower,SPIRE_500.prior_flux_lower],
              'f_up_lim':[SPIRE_250.prior_flux_upper,SPIRE_350.prior_flux_upper,SPIRE_500.prior_flux_upper],
              'bkg_prior':[SPIRE_250.bkg[0],SPIRE_350.bkg[0],SPIRE_500.bkg[0]],
              'bkg_prior_sig':[SPIRE_250.bkg[1],SPIRE_350.bkg[1],SPIRE_500.bkg[1]],
          'npix_psw':SPIRE_250.snpix,
          'nnz_psw':SPIRE_250.amat_data.size,
          'db_psw':SPIRE_250.sim,
          'sigma_psw':SPIRE_250.snim,
          'Val_psw':SPIRE_250.amat_data,
          'Row_psw': SPIRE_250.amat_row.astype(long),
          'Col_psw': SPIRE_250.amat_col.astype(long),
          'npix_pmw':SPIRE_350.snpix,
          'nnz_pmw':SPIRE_350.amat_data.size,
          'db_pmw':SPIRE_350.sim,
          'sigma_pmw':SPIRE_350.snim,
          'Val_pmw':SPIRE_350.amat_data,
          'Row_pmw': SPIRE_350.amat_row.astype(long),
          'Col_pmw': SPIRE_350.amat_col.astype(long),
          'npix_plw':SPIRE_500.snpix,
          'nnz_plw':SPIRE_500.amat_data.size,
          'db_plw':SPIRE_500.sim,
          'sigma_plw':SPIRE_500.snim,
          'Val_plw':SPIRE_500.amat_data,
          'Row_plw': SPIRE_500.amat_row.astype(long),
          'Col_plw': SPIRE_500.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+logSPIRE'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit