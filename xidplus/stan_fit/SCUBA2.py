__author__ = 'pdh21'
import os
output_dir=os.getcwd()
import pystan
import pickle
import inspect
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

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
              'Col': prior.amat_col.astype(long),
              'f_low_lim_sc2': SCUBA2.prior_flux_lower,
              'f_up_lim_sc2': SCUBA2.prior_flux_upper}

    #see if model has already been compiled. If not, compile and save it
    import os
    model_file="./XID+SCUBA2.pkl"
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
        sm = pystan.StanModel(file=stan_path+'XID+SCUBA2.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
        if optimise is True:
                fit=sm.optimizing(data=XID_data)
        else:
            fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    #return fit data
    return fit
