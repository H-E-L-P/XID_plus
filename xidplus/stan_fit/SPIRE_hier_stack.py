stan_path='../../stan_models/'

import pystan
import pickle

def LBG_highz(prior,chains=4,iter=1000):
    """Fit single SPIRE map with hierarchical stacking population"""



    #input data into a dictionary

    XID_data={'npix':prior.snpix,
          'nsrc':prior.nsrc,
          'nsrc_z':prior.stack_nsrc,
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
    model_file="./XID+highz.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)
            fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file=stan_path+'XID+highz.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    #return fit data
    return fit
