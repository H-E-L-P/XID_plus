import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import XIDp_mod_beta as xid_mod
import os
import sys

#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/'

infile=output_folder+'Lacey_rbandcut_19_8_log_flux.pkl'
with open(infile, "rb") as f:
    dictname = pickle.load(f)
prior250=dictname['psw']
prior350=dictname['pmw']    
prior500=dictname['plw']

posterior=dictname['posterior']
posterior.stan_fit=np.power(10.0,posterior.stan_fit)
hdulist=xid_mod.create_XIDp_SPIREcat(posterior,prior250,prior350,prior500)
hdulist.writeto(output_folder+'Lacey_rbandcut_19_8_flux.fits')
