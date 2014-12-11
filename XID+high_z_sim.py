import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import XIDp_mod_beta as xid_mod
import pickle
import dill
import sys

output_folder='/research/astro/fir/HELP/XID_plus_output/sims/'

mu_sim=sys.argv[1]
sig_sim=sys.argv[2]
outfile=output_folder+'goodss_highz_fit_250_sim_prior_'+str(mu_sim)+'_'+str(sig_sim)+'.pkl'
tempf=pickle.load(open(outfile,'rb'))
prior250=tempf['prior250']
fit_data,chains,iter=xid_mod.lstdrv_stan_highz(prior250)


# In[ ]:

outfile=output_folder+'goodss_highz_fit_250_sim_post_'+str(mu_sim)+'_'+str(sig_sim)+'.pkl'
with open(outfile, 'wb') as f:
            pickle.dump({'fit':fit_data}, f)
