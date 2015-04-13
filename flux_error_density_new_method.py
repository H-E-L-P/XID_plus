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
hdulist=fits.open(folder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise__PSWXID_S100_50mic_test.fits')
fcat=hdulist[1].data
ind=fcat['xid'] >= 0
fcat=fcat[ind]
hdulist.close()

#---Read in truth catalogue---
folder='/research/astro/fir/cclarke/lacey/released/'
hdulist=fits.open(folder+'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test.fits')
fcat_sim=hdulist[1].data
hdulist.close()

#---Read in XID+ catalogue---
#folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'
folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/'


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
#output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'

infile=folder+'Tiled_master_Lacey_notlog_flux.pkl'
with open(infile, "rb") as f:
    obj = pickle.load(f)
prior250=obj['psw']
prior350=obj['pmw']    
prior500=obj['plw']

posterior=obj['posterior']

samples,chains,params=posterior.stan_fit.shape
flattened_post=posterior.stan_fit.reshape(samples*chains,params)
nsources_xidp=idx_xidp.size
nsources_xid=idx.size

error_percent_xidp_250=np.empty((nsources_xidp))
error_percent_xid_250=np.empty((nsources_xid))

import scipy.stats as stats
for i in range(0,nsources_xidp):
    error_percent_xidp_250[i]=stats.percentileofscore(flattened_post[:,i],fcat_sim['S250'][idx_xidp][i])

for i in range(0,nsources_xid):
    error_percent_xid_250[i]=100.0*stats.norm.cdf(fcat_sim['S250'][idx][i],loc=fcat['F250'][i],scale=fcat['E250'][i])

bins=np.array([0.0,0.0032,0.135,2.275,15.87,50.0,84.13,97.725,99.865,99.9968,100.0])
hist_xidp,bin_eges_xidp=np.histogram(error_percent_xidp_250,bins)
hist_xid,bin_eges_xid=np.histogram(error_percent_xid_250,bins)
sigma=np.arange(-4.5,5.0,1)

plt.plot(sigma,hist_xid/float(nsources_xid), label='DESPHOT')
plt.plot(sigma,hist_xidp/float(nsources_xidp), 'r',label='XID+')
print hist_xid.sum(),hist_xidp.sum()
plt.xlabel('Flux error density (Sigma)')

plt.legend()
plt.savefig("error_density_flux_test.pdf")
