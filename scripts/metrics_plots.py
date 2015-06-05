#---import modules---
from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use('PDF')
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('/research/astro/fir/HELP/XID_plus/')
import XIDp_mod_beta
import pickle
pdf_pages=PdfPages("error_density_flux_test_uninform.pdf")

#---Read in DESPHOT catalogue---
#folder='/research/astro/fir/HELP/DESPHOT/'
#hdulist=fits.open(folder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise__PSWXID_S100_50mic_test.fits')
#fcat=hdulist[1].data
#ind=fcat['xid'] >= 0
#fcat=fcat[ind]
#hdulist.close()

#---Read in truth catalogue---
folder='/research/astro/fir/cclarke/lacey/released/'
hdulist=fits.open(folder+'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test.fits')
fcat_sim=hdulist[1].data
hdulist.close()

fcat_sim=fcat_sim[fcat_sim['S100']>0.050]

#---match DESPHOT and real catalogues---
#from astropy.coordinates import SkyCoord
#from astropy import units as u
#c= SkyCoord(ra=fcat['INRA']*u.degree,dec=fcat['INDEC']*u.degree)
#c1=SkyCoord(ra=fcat_sim['RA']*u.degree,dec=fcat_sim['DEC']*u.degree)
#idx,d2d,d3d,= c.match_to_catalog_sky(c1)

idx_xidp=fcat_sim['S100'] >0.050#cut so that only sources with a 100micron flux of > 50 micro janskys (Roseboom et al. 2010 cut 24 micron sources at 50microJys)
idx_xidpT=fcat_sim['S100'] >0.050#cut so that only sources with a 100micron flux of > 50 micro janskys (Roseboom et al. 2010 cut 24 micron sources at 50microJys)


#---Read in XID+ posterior---

#folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'
folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/'
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
#nsources_xid=idx.size



error_percent_xidp_250=np.empty((nsources_xidp))
IQR_xidp_250=np.empty((nsources_xidp))
accuracy_xidp_250=np.empty((nsources_xidp))
#error_percent_xid_250=np.empty((nsources_xid))

import scipy.stats as stats
for i in range(0,nsources_xidp):
    error_percent_xidp_250[i]=stats.percentileofscore(flattened_post[:,i],fcat_sim['S250'][idx_xidp][i])
    IQR_xidp_250[i]=np.subtract(*np.percentile(flattened_post[:,i],[75.0,25.0]))/np.median(flattened_post[:,i])
    accuracy_xidp_250[i]=(np.median(flattened_post[:,i])-fcat_sim['S250'][idx_xidp][i])/fcat_sim['S250'][idx_xidp][i]

#for i in range(0,nsources_xid):
#    error_percent_xid_250[i]=100.0*stats.norm.cdf(fcat_sim['S250'][idx][i],loc=fcat['F250'][i],scale=fcat['E250'][i])
ind_3mjy=fcat_sim['S250'][idx_xidp]>3

bins=np.array([0.0,0.0032,0.135,2.275,15.87,50.0,84.13,97.725,99.865,99.9968,100.0])
hist_xidp,bin_eges_xidp=np.histogram(error_percent_xidp_250[ind_3mjy],bins)
#hist_xid,bin_eges_xid=np.histogram(error_percent_xid_250,bins)
sigma=np.arange(-4.5,5.0,1)
fig=plt.figure()
#plt.plot(sigma,hist_xid/float(nsources_xid), label='DESPHOT')
plt.plot(sigma,hist_xidp/float(ind_3mjy.sum()), 'r',label='XID+')
plt.xlabel('Flux error density (Sigma)')

plt.legend()
pdf_pages.savefig(fig)

fig2=plt.figure()
#H,xedge,yedge=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),error_percent_xidp_250,bins=[np.arange(-3,3,0.25),bins])
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),error_percent_xidp_250,bins=[np.arange(0.5,2.2,0.01),bins])
plt.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto')
plt.xlabel('S250 mJy')
plt.ylabel('z score')
print H.shape,xedges.shape,yedges.shape
plt.colorbar()
pdf_pages.savefig(fig2)
#bins=[np.arange(-3,3,0.25),bin_eges_xidp]


#flux precision
fig3=plt.figure()
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),IQR_xidp_250,bins=[np.arange(0.5,2.2,0.01),np.arange(0,2,0.05)])
plt.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto')
plt.xlabel('S250 mJy')
plt.ylabel('IQR mJy/True')
plt.colorbar()
pdf_pages.savefig(fig3)

fig4=plt.figure()
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),accuracy_xidp_250,bins=[np.arange(0.5,2.2,0.01),np.arange(-2,2,0.01)])
plt.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto')
plt.xlabel('S250 mJy')
plt.ylabel('Obs -True / True mJy')
plt.colorbar()
pdf_pages.savefig(fig4)

pdf_pages.close()
