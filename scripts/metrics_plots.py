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
folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/old/'
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
error_percent_xidp_350=np.empty((nsources_xidp))
IQR_xidp_350=np.empty((nsources_xidp))
accuracy_xidp_350=np.empty((nsources_xidp))
error_percent_xidp_500=np.empty((nsources_xidp))
IQR_xidp_500=np.empty((nsources_xidp))
accuracy_xidp_500=np.empty((nsources_xidp))
#error_percent_xid_250=np.empty((nsources_xid))

import scipy.stats as stats
for i in range(0,nsources_xidp):
    #---250---
    error_percent_xidp_250[i]=stats.percentileofscore(flattened_post[:,i],fcat_sim['S250'][idx_xidp][i])
    IQR_xidp_250[i]=np.subtract(*np.percentile(flattened_post[:,i],[75.0,25.0]))/np.median(flattened_post[:,i])
    accuracy_xidp_250[i]=(np.median(flattened_post[:,i])-fcat_sim['S250'][idx_xidp][i])/fcat_sim['S250'][idx_xidp][i]
    #-----350
    error_percent_xidp_350[i]=stats.percentileofscore(flattened_post[:,i+prior250.nsrc+1],fcat_sim['S350'][idx_xidp][i])
    IQR_xidp_350[i]=np.subtract(*np.percentile(flattened_post[:,i+prior250.nsrc+1],[75.0,25.0]))/np.median(flattened_post[:,i+prior250.nsrc+1])
    accuracy_xidp_350[i]=(np.median(flattened_post[:,i+prior250.nsrc+1])-fcat_sim['S350'][idx_xidp][i])/fcat_sim['S350'][idx_xidp][i]
    #---500---
    error_percent_xidp_500[i]=stats.percentileofscore(flattened_post[:,2*prior250.nsrc+2+i],fcat_sim['S500'][idx_xidp][i])
    IQR_xidp_500[i]=np.subtract(*np.percentile(flattened_post[:,2*prior250.nsrc+2+i],[75.0,25.0]))/np.median(flattened_post[:,2*prior250.nsrc+2+i])
    accuracy_xidp_500[i]=(np.median(flattened_post[:,2*prior250.nsrc+2+i])-fcat_sim['S500'][idx_xidp][i])/fcat_sim['S500'][idx_xidp][i]
#for i in range(0,nsources_xid):
#    error_percent_xid_250[i]=100.0*stats.norm.cdf(fcat_sim['S250'][idx][i],loc=fcat['F250'][i],scale=fcat['E250'][i])
ind_3mjy_250=fcat_sim['S250'][idx_xidp]>3
ind_3mjy_350=fcat_sim['S350'][idx_xidp]>3
ind_3mjy_500=fcat_sim['S500'][idx_xidp]>3

bins=np.array([0.0,0.0032,0.135,2.275,15.87,50.0,84.13,97.725,99.865,99.9968,100.0])
#hist_xid,bin_eges_xid=np.histogram(error_percent_xid_250,bins)
sigma=np.arange(-4.5,5.0,1)
fig,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,sharey=True,figsize=(10,20))
hist_xidp,bin_eges_xidp=np.histogram(error_percent_xidp_250[ind_3mjy_250],bins)
ax1.plot(sigma,hist_xidp/float(ind_3mjy_250.sum()), 'r',label=r'XID+ ')
hist_xidp,bin_eges_xidp=np.histogram(error_percent_xidp_350[ind_3mjy_350],bins)
ax2.plot(sigma,hist_xidp/float(ind_3mjy_350.sum()), 'r',label=r'XID+ ')
hist_xidp,bin_eges_xidp=np.histogram(error_percent_xidp_500[ind_3mjy_350],bins)
ax3.plot(sigma,hist_xidp/float(ind_3mjy_500.sum()), 'r',label=r'XID+ ')
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]],visible=False)
ax3.set_xlabel('Flux error density (Sigma)')
fig.subplots_adjust(hspace=0)


plt.legend()
pdf_pages.savefig(fig)

fig2,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,sharey=True,figsize=(10,20))
#H,xedge,yedge=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),error_percent_xidp_250,bins=[np.arange(-3,3,0.25),bins])
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),error_percent_xidp_250,bins=[np.arange(0.5,2.2,0.01),bins])
im=ax1.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S350'][idx_xidp]),error_percent_xidp_350,bins=[np.arange(0.5,2.2,0.01),bins])
im=ax2.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S500'][idx_xidp]),error_percent_xidp_500,bins=[np.arange(0.5,2.2,0.01),bins])
im=ax3.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)

ax3.set_xlabel('Flux mJy')
ax2.set_ylabel('z score')
fig2.subplots_adjust(hspace=0)

fig2.subplots_adjust(right=0.8)
cbar_ax=fig2.add_axes([0.85,0.15,0.05,0.7])
fig2.colorbar(im,cax=cbar_ax)
pdf_pages.savefig(fig2)
#bins=[np.arange(-3,3,0.25),bin_eges_xidp]


#flux precision
fig3, (ax1,ax2,ax3)=plt.subplots(3,sharex=True,sharey=True,figsize=(10,20))
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),IQR_xidp_250,bins=[np.arange(0.5,2.2,0.01),np.arange(0,2,0.05)])
ax1.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S350'][idx_xidp]),IQR_xidp_350,bins=[np.arange(0.5,2.2,0.01),np.arange(0,2,0.05)])
ax2.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S500'][idx_xidp]),IQR_xidp_500,bins=[np.arange(0.5,2.2,0.01),np.arange(0,2,0.05)])
ax3.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)
ax3.set_xlabel('Flux mJy')
ax2.set_ylabel('IQR/True Flux')
fig3.subplots_adjust(hspace=0)


fig3.subplots_adjust(right=0.8)
cbar_ax=fig3.add_axes([0.85,0.15,0.05,0.7])
fig3.colorbar(im,cax=cbar_ax)
pdf_pages.savefig(fig3)

fig4, (ax1,ax2,ax3)=plt.subplots(3,sharex=True,sharey=True,figsize=(10,20))
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),accuracy_xidp_250,bins=[np.arange(0.5,2.2,0.01),np.arange(-2,2,0.01)])
ax1.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S350'][idx_xidp]),accuracy_xidp_350,bins=[np.arange(0.5,2.2,0.01),np.arange(-2,2,0.01)])
ax2.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S500'][idx_xidp]),accuracy_xidp_500,bins=[np.arange(0.5,2.2,0.01),np.arange(-2,2,0.01)])
im=ax3.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto',vmin=0,vmax=1.0)

ax3.set_xlabel('S250 mJy')
ax2.set_ylabel('Obs -True / True mJy')
fig4.subplots_adjust(hspace=0)

fig4.subplots_adjust(right=0.8)
cbar_ax=fig4.add_axes([0.85,0.15,0.05,0.7])
fig4.colorbar(im,cax=cbar_ax)
pdf_pages.savefig(fig4)


fig5,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,sharey=True,figsize=(10,20))
#H,xedge,yedge=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),error_percent_xidp_250,bins=[np.arange(-3,3,0.25),bins])
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),error_percent_xidp_250,bins=[np.arange(0.5,2.2,0.01),bins])
im=ax1.imshow(H.T,interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto')
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S350'][idx_xidp]),error_percent_xidp_350,bins=[np.arange(0.5,2.2,0.01),bins])
im=ax2.imshow(H.T,interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto')
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S500'][idx_xidp]),error_percent_xidp_500,bins=[np.arange(0.5,2.2,0.01),bins])
im=ax3.imshow(H.T,interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto')

ax3.set_xlabel('Flux mJy')
ax2.set_ylabel('z score')
fig5.subplots_adjust(hspace=0)

fig5.subplots_adjust(right=0.8)
cbar_ax=fig5.add_axes([0.85,0.15,0.05,0.7])
fig5.colorbar(im,cax=cbar_ax)
pdf_pages.savefig(fig5)


pdf_pages.close()
