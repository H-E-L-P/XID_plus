__author__ = 'pdh21'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def all_bands(PACS_100,PACS_160,chains=4,iter=1000,optimise=False):
    """Fit all three SPIRE maps using stan"""


    #input data into a dictionary

    XID_data={'nsrc':PACS_100.nsrc,
          'npix_psw':PACS_100.snpix,
          'nnz_psw':PACS_100.amat_data.size,
          'db_psw':PACS_100.sim,
          'sigma_psw':PACS_100.snim,
          'bkg_prior_psw':PACS_100.bkg[0],
          'bkg_prior_sig_psw':PACS_100.bkg[1],
          'Val_psw':PACS_100.amat_data,
          'Row_psw': PACS_100.amat_row.astype(long),
          'Col_psw': PACS_100.amat_col.astype(long),
          'f_low_lim_psw': PACS_100.prior_flux_lower,
          'f_up_lim_psw': PACS_100.prior_flux_upper,
          'npix_pmw':PACS_160.snpix,
          'nnz_pmw':PACS_160.amat_data.size,
          'db_pmw':PACS_160.sim,
          'sigma_pmw':PACS_160.snim,
          'bkg_prior_pmw':PACS_160.bkg[0],
          'bkg_prior_sig_pmw':PACS_160.bkg[1],
          'Val_pmw':PACS_160.amat_data,
          'Row_pmw': PACS_160.amat_row.astype(long),
          'Col_pmw': PACS_160.amat_col.astype(long),
          'f_low_lim_pmw': PACS_160.prior_flux_lower,
          'f_up_lim_pmw': PACS_160.prior_flux_upper}

    #see if model has already been compiled. If not, compile and save it
    model_file=output_dir+"/XID+PACS.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)

            
       fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True,init='random')
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file=stan_path+'XID+PACS.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
           
            
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True,init='random')
    #return fit data
    return fit
