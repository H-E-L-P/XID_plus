import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import sys

sys.path.append('/research/astro/fir/HELP/XID_plus/')

import XIDp_mod_beta as xid_mod
import os

folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'

infile=folder+'Tiled_master_Lacey_notlog_flux_norm1.pkl'
with open(infile, "rb") as f:
    obj = pickle.load(f)
prior250=obj['psw']
prior350=obj['pmw']    
prior500=obj['plw']

posterior=obj['posterior']

thdulist_master=xid_mod.create_XIDp_SPIREcat_nocov(posterior,prior250,prior350,prior500)
thdulist_master.writeto(folder+'Tiled_SPIRE_cat_flux_notlog_norm1.fits')

