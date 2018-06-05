from numpy import long

__author__ = 'pdh21'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)
import numpy as np

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'

def MIPS_PACS_SPIRE(phot_priors,sed_prior_model,chains=4,iter=1000,max_treedepth=10):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior100=phot_priors[1]
    prior160=phot_priors[2]
    prior250=phot_priors[3]
    prior350=phot_priors[4]
    prior500=phot_priors[5]

    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior24.bkg[0], prior100.bkg[0],
                      prior160.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior24.bkg[1], prior100.bkg[1],
                          prior160.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.0001,0.1, 0.1, 0.1, 0.1, 0.1],
        'z_median': prior24.z_median,
        'z_sig': prior24.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'npix_mips24': prior24.snpix,
        'nnz_mips24': prior24.amat_data.size,
        'db_mips24': prior24.sim,
        'sigma_mips24': prior24.snim,
        'Val_mips24': prior24.amat_data,
        'Row_mips24': prior24.amat_row.astype(np.long),
        'Col_mips24': prior24.amat_col.astype(np.long),
        'npix_pacs100': prior100.snpix,
        'nnz_pacs100': prior100.amat_data.size,
        'db_pacs100': prior100.sim,
        'sigma_pacs100': prior100.snim,
        'Val_pacs100': prior100.amat_data,
        'Row_pacs100': prior100.amat_row.astype(np.long),
        'Col_pacs100': prior100.amat_col.astype(np.long),
        'npix_pacs160': prior160.snpix,
        'nnz_pacs160': prior160.amat_data.size,
        'db_pacs160': prior160.sim,
        'sigma_pacs160': prior160.snim,
        'Val_pacs160': prior160.amat_data,
        'Row_pacs160': prior160.amat_row.astype(np.long),
        'Col_pacs160': prior160.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+IR_SED'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True,control=dict(max_treedepth=max_treedepth))
    #return fit data
    return fit

def PACS_SPIRE(phot_priors,sed_prior_model,chains=4,iter=1000,max_treedepth=10):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior100=phot_priors[0]
    prior160=phot_priors[1]
    prior250=phot_priors[2]
    prior350=phot_priors[3]
    prior500=phot_priors[4]

    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior100.bkg[0],
                      prior160.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior100.bkg[1],
                          prior160.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.1, 0.1, 0.1, 0.1, 0.1],
        'z_median': prior100.z_median,
        'z_sig': prior100.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'npix_pacs100': prior100.snpix,
        'nnz_pacs100': prior100.amat_data.size,
        'db_pacs100': prior100.sim,
        'sigma_pacs100': prior100.snim,
        'Val_pacs100': prior100.amat_data,
        'Row_pacs100': prior100.amat_row.astype(np.long),
        'Col_pacs100': prior100.amat_col.astype(np.long),
        'npix_pacs160': prior160.snpix,
        'nnz_pacs160': prior160.amat_data.size,
        'db_pacs160': prior160.sim,
        'sigma_pacs160': prior160.snim,
        'Val_pacs160': prior160.amat_data,
        'Row_pacs160': prior160.amat_row.astype(np.long),
        'Col_pacs160': prior160.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+Herschel_SED'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True,control=dict(max_treedepth=max_treedepth))
    #return fit data
    return fit

def MIPS_SPIRE(phot_priors,sed_prior_model,chains=4,iter=1000,max_treedepth=10):

    """
    Fit the three SPIRE bands

    :param priors: list of xidplus.prior class objects. Order (MIPS,PACS100,PACS160,SPIRE250,SPIRE350,SPIRE500)
    :param sed_prior: xidplus.sed.sed_prior class
    :param chains: number of chains
    :param iter: number of iterations
    :return: pystan fit object
    """
    prior24=phot_priors[0]
    prior250=phot_priors[1]
    prior350=phot_priors[2]
    prior500=phot_priors[3]

    #input data into a dictionary
    XID_data = {
        'nsrc': prior250.nsrc,
        'bkg_prior': [prior24.bkg[0],prior250.bkg[0], prior350.bkg[0], prior500.bkg[0]],
        'bkg_prior_sig': [prior24.bkg[1],prior250.bkg[1], prior350.bkg[1], prior500.bkg[1]],
        'conf_prior_sig': [0.0001, 0.1, 0.1, 0.1],
        'z_median': prior24.z_median,
        'z_sig': prior24.z_sig,
        'npix_psw': prior250.snpix,
        'nnz_psw': prior250.amat_data.size,
        'db_psw': prior250.sim,
        'sigma_psw': prior250.snim,
        'Val_psw': prior250.amat_data,
        'Row_psw': prior250.amat_row.astype(np.long),
        'Col_psw': prior250.amat_col.astype(np.long),
        'npix_pmw': prior350.snpix,
        'nnz_pmw': prior350.amat_data.size,
        'db_pmw': prior350.sim,
        'sigma_pmw': prior350.snim,
        'Val_pmw': prior350.amat_data,
        'Row_pmw': prior350.amat_row.astype(np.long),
        'Col_pmw': prior350.amat_col.astype(np.long),
        'npix_plw': prior500.snpix,
        'nnz_plw': prior500.amat_data.size,
        'db_plw': prior500.sim,
        'sigma_plw': prior500.snim,
        'Val_plw': prior500.amat_data,
        'Row_plw': prior500.amat_row.astype(np.long),
        'Col_plw': prior500.amat_col.astype(np.long),
        'npix_mips24': prior24.snpix,
        'nnz_mips24': prior24.amat_data.size,
        'db_mips24': prior24.sim,
        'sigma_mips24': prior24.snim,
        'Val_mips24': prior24.amat_data,
        'Row_mips24': prior24.amat_row.astype(np.long),
        'Col_mips24': prior24.amat_col.astype(np.long),
        'nTemp': sed_prior_model.shape[0],
        'nz': sed_prior_model.shape[2],
        'nband': sed_prior_model.shape[1],
        'SEDs': sed_prior_model,
    }


    #see if model has already been compiled. If not, compile and save it
    model_file='/XID+MIPS_SPIRE_SED'
    from xidplus.stan_fit import get_stancode
    sm = get_stancode(model_file)


    fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True,control=dict(max_treedepth=max_treedepth))
    #return fit data
    return fit