__author__ = 'pdh21'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
def MIPS_24(MIPS_24,chains=4,iter=1000,optimise=False):
    """Fit MIPS 24 maps using stan"""


    #input data into a dictionary

    XID_data={'nsrc':MIPS_24.nsrc,
          'npix_psw':MIPS_24.snpix,
          'nnz_psw':MIPS_24.amat_data.size,
          'db_psw':MIPS_24.sim,
          'sigma_psw':MIPS_24.snim,
          'bkg_prior_psw':MIPS_24.bkg[0],
          'bkg_prior_sig_psw':MIPS_24.bkg[1],
          'Val_psw':MIPS_24.amat_data,
          'Row_psw': MIPS_24.amat_row.astype(long),
          'Col_psw': MIPS_24.amat_col.astype(long),
          'f_low_lim_psw': MIPS_24.prior_flux_lower,
          'f_up_lim_psw': MIPS_24.prior_flux_upper}

    #see if model has already been compiled. If not, compile and save it
    model_file=output_dir+"/XID+MIPS.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)
       if optimise is True:
           fit=sm.optimizing(data=XID_data,iter=3000)
       else: 
           fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file=stan_path+'XID+MIPS.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
           
        if optimise is True:
            fit=sm.optimizing(data=XID_data,iter=3000)
        else:
            fit = sm.sampling(data=XID_data,iter=iter,chains=chains,verbose=True)
    #return fit data
    return fit
