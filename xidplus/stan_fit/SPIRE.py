__author__ = 'pdh21'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def all_bands(SPIRE_250,SPIRE_350,SPIRE_500,chains=4,iter=1000,optimise=False):
    """Fit all three SPIRE maps using stan"""


    #input data into a dictionary

    XID_data={'nsrc':SPIRE_250.nsrc,
          'npix_psw':SPIRE_250.snpix,
          'nnz_psw':SPIRE_250.amat_data.size,
          'db_psw':SPIRE_250.sim,
          'sigma_psw':SPIRE_250.snim,
          'bkg_prior_psw':SPIRE_250.bkg[0],
          'bkg_prior_sig_psw':SPIRE_250.bkg[1],
          'Val_psw':SPIRE_250.amat_data,
          'Row_psw': SPIRE_250.amat_row.astype(long),
          'Col_psw': SPIRE_250.amat_col.astype(long),
          'f_low_lim_psw': SPIRE_250.prior_flux_lower,
          'f_up_lim_psw': SPIRE_250.prior_flux_upper,
          'npix_pmw':SPIRE_350.snpix,
          'nnz_pmw':SPIRE_350.amat_data.size,
          'db_pmw':SPIRE_350.sim,
          'sigma_pmw':SPIRE_350.snim,
          'bkg_prior_pmw':SPIRE_350.bkg[0],
          'bkg_prior_sig_pmw':SPIRE_350.bkg[1],
          'Val_pmw':SPIRE_350.amat_data,
          'Row_pmw': SPIRE_350.amat_row.astype(long),
          'Col_pmw': SPIRE_350.amat_col.astype(long),
          'f_low_lim_pmw': SPIRE_350.prior_flux_lower,
          'f_up_lim_pmw': SPIRE_350.prior_flux_upper,
          'npix_plw':SPIRE_500.snpix,
          'nnz_plw':SPIRE_500.amat_data.size,
          'db_plw':SPIRE_500.sim,
          'sigma_plw':SPIRE_500.snim,
          'bkg_prior_plw':SPIRE_500.bkg[0],
          'bkg_prior_sig_plw':SPIRE_500.bkg[1],
          'Val_plw':SPIRE_500.amat_data,
          'Row_plw': SPIRE_500.amat_row.astype(long),
          'Col_plw': SPIRE_500.amat_col.astype(long),
          'f_low_lim_plw': SPIRE_500.prior_flux_lower,
          'f_up_lim_plw': SPIRE_500.prior_flux_upper}

    #see if model has already been compiled. If not, compile and save it
    model_file=output_dir+"/XID+SPIRE.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)

            
       fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file=stan_path+'XID+SPIRE.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
           
            
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit

def single_band(prior,chains=4,iter=1000):
    """Fit single SPIRE map using stan"""


    #input data into a dictionary

    XID_data={'npix':prior.snpix,
          'nsrc':prior.nsrc,
          'nnz':prior.amat_data.size,
          'db':prior.sim,
          'sigma':prior.snim,
          'bkg_prior':prior.bkg[0],
          'bkg_prior_sig':prior.bkg[1],
          'Val':prior.amat_data,
          'Row': prior.amat_row.astype(long),
          'Col': prior.amat_col.astype(long)}

    #see if model has already been compiled. If not, compile and save it
    import os
    model_file="./XID+_basic.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)
            if optimise is True:
                fit=sm.optimizing(data=XID_data)
            else:
                fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file=stan_path+'XIDfit.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
        if optimise is True:
                fit=sm.optimizing(data=XID_data)
        else:
            fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    #return fit data
    return fit

def flux_prior_all_bands(SPIRE_250,SPIRE_350,SPIRE_500,chains=4,iter=1000):
    """Fit all three SPIRE maps using stan"""


    #input data into a dictionary

    XID_data={'nsrc':SPIRE_250.nsrc,
          'npix_psw':SPIRE_250.snpix,
          'nnz_psw':SPIRE_250.amat_data.size,
          'db_psw':SPIRE_250.sim,
          'sigma_psw':SPIRE_250.snim,
          'bkg_prior_psw':SPIRE_250.bkg[0],
          'bkg_prior_sig_psw':SPIRE_250.bkg[1],
          'Val_psw':SPIRE_250.amat_data,
          'Row_psw': SPIRE_250.amat_row.astype(long),
          'Col_psw': SPIRE_250.amat_col.astype(long),
          'psw_prior': SPIRE_250.sflux,
          'npix_pmw':SPIRE_350.snpix,
          'nnz_pmw':SPIRE_350.amat_data.size,
          'db_pmw':SPIRE_350.sim,
          'sigma_pmw':SPIRE_350.snim,
          'bkg_prior_pmw':SPIRE_350.bkg[0],
          'bkg_prior_sig_pmw':SPIRE_350.bkg[1],
          'Val_pmw':SPIRE_350.amat_data,
          'Row_pmw': SPIRE_350.amat_row.astype(long),
          'Col_pmw': SPIRE_350.amat_col.astype(long),
          'pmw_prior': SPIRE_350.sflux,
          'npix_plw':SPIRE_500.snpix,
          'nnz_plw':SPIRE_500.amat_data.size,
          'db_plw':SPIRE_500.sim,
          'sigma_plw':SPIRE_500.snim,
          'bkg_prior_plw':SPIRE_500.bkg[0],
          'bkg_prior_sig_plw':SPIRE_500.bkg[1],
          'Val_plw':SPIRE_500.amat_data,
          'Row_plw': SPIRE_500.amat_row.astype(long),
          'Col_plw': SPIRE_500.amat_col.astype(long),
          'plw_prior': SPIRE_500.sflux}

    #see if model has already been compiled. If not, compile and save it
    import os
    model_file="./XID+SPIRE_prior.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)
            fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file=stan_path+'XID+SPIRE_prior.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    #return fit data
    return fit
