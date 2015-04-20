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
folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'
#folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/'


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
flattened_post=np.log10(posterior.stan_fit.reshape(samples*chains,params))

bins=np.arange(-6,4,0.1)
hist_sim,bin_edges=np.histogram(np.log10(fcat_sim['S250'][idx_xidp]),bins=bins)

fig=plt.figure(figsize=(20,20))
hist_post=np.empty((hist_sim.size,flattened_post.shape[0]))
#hist_post_pT=np.empty((hist_sim.size,flattened_post_pT.shape[0]))

print bins.size,hist_sim.size
plt.plot(bins[1:],hist_sim, label='Truth')
for i in np.arange(0,flattened_post.shape[0]):
    hist_samp,bin_edges=np.histogram(flattened_post[i,0:prior250.nsrc],bins=bins)
    hist_post[:,i]=hist_samp
    #hist_samp_pT,bin_edges_pT=np.histogram(flattened_post_pT[i,0:prior250_pT.nsrc],bins=bins)
    #hist_post_pT[:,i]=hist_samp_pT    
    #plt.plot(bins[1:],hist_post,'g',alpha=0.5)
plt.plot(bins[1:],np.mean(hist_post,axis=1),'g',label='XID+ mean')
plt.fill_between(bins[1:],np.percentile(hist_post,84,axis=1),np.percentile(hist_post,16,axis=1),facecolor='green',alpha=0.2,label='XID+ 84th-16th percentile')
#plt.plot(bins[1:],np.mean(hist_post_pT,axis=1),'r',label='XID+Tiling mean')
#plt.fill_between(bins[1:],np.percentile(hist_post_pT,84,axis=1),np.percentile(hist_post_pT,16,axis=1),facecolor='red',alpha=0.2,label='XID+Tiling 84th-16th percentile')
plt.xlabel('$\log_{10}S_{250\mathrm{\mu m}} (\mathrm{mJy})$')
plt.legend()
plt.savefig("flux_distribution)flux_prior.pdf")
