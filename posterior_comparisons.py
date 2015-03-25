#---import modules---
from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use('PDF')
import pylab as plt
import sys
sys.path.append('/research/astro/fir/HELP/XID_plus/')
import XIDp_mod_beta
import pickle

#---Read in DESPHOT catalogue---
folder='/research/astro/fir/HELP/DESPHOT/'
hdulist=fits.open(folder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise__PSWXID_S100_50mic.fits')
fcat=hdulist[1].data
ind=fcat['xid'] >= 0
fcat=fcat[ind]
hdulist.close()

#---Read in truth catalogue---
folder='/research/astro/fir/cclarke/lacey/released/'
hdulist=fits.open(folder+'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS.fits')
fcat_sim=hdulist[1].data
hdulist.close()

#---Read in XID+ catalogue---
folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior/'
hdulist=fits.open(folder+'Tiled_SPIRE_cat_flux_notlog.fits')
fcat_xidp=hdulist[1].data
hdulist.close()

#---match DESPHOT and real catalogues---
from astropy.coordinates import SkyCoord
from astropy import units as u
c= SkyCoord(ra=fcat['INRA']*u.degree,dec=fcat['INDEC']*u.degree)
c1=SkyCoord(ra=fcat_sim['RA']*u.degree,dec=fcat_sim['DEC']*u.degree)
idx,d2d,d3d,= c.match_to_catalog_sky(c1)

#---match XID+ and real catalogues---
c= SkyCoord(ra=fcat_xidp['ra']*u.degree,dec=fcat_xidp['dec']*u.degree)
c1=SkyCoord(ra=fcat_sim['RA']*u.degree,dec=fcat_sim['DEC']*u.degree)
idx_xidp,d2d,d3d,= c.match_to_catalog_sky(c1)

#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior/'

infile=output_folder+'Tiled_master_Lacey_notlog_flux.pkl'
with open(infile, "rb") as f:
    obj = pickle.load(f)
prior250=obj['psw']
prior350=obj['pmw']    
prior500=obj['plw']

posterior=obj['posterior']

samples,chains,params=posterior.stan_fit.shape
flattened_post=np.log10(posterior.stan_fit.reshape(samples*chains,params))

import triangle

figure = triangle.corner(flattened_post[:,[465,533]])
figure.savefig("triangle.pdf")
