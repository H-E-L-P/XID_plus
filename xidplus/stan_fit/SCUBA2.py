from numpy import long

__author__ = 'pdh21-mwls2'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def single_band(SCUBA_BAND,chains=4,iter=1000):

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

    XID_data={'nsrc':SCUBA_BAND.nsrc,
              'f_low_lim':[SCUBA_BAND.prior_flux_lower],
              'f_up_lim':[SCUBA_BAND.prior_flux_upper],
              'bkg_prior':[SCUBA_BAND.bkg[0]],
              'bkg_prior_sig':[SCUBA_BAND.bkg[1]],
          'npix_S2b':SCUBA_BAND.snpix,
          'nnz_S2b':SCUBA_BAND.amat_data.size,
          'db_S2b':SCUBA_BAND.sim,
          'sigma_S2b':SCUBA_BAND.snim,
          'Val_S2b':SCUBA_BAND.amat_data,
          'Row_S2b': SCUBA_BAND.amat_row.astype(long),
          'Col_S2b': SCUBA_BAND.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+SCUBA2'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit
