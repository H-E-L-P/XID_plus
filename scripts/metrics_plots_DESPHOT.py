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
pdf_pages=PdfPages("error_density_flux_test_DESPHOT.pdf")

#---Read in DESPHOT catalogue---
folder='/research/astro/fir/HELP/DESPHOT/'
hdulist=fits.open(folder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise__PSWXID_S100_50mic_test.fits')
fcat=hdulist[1].data
nsources_xid=fcat.shape[0]
print nsources_xid
hdulist.close()

#---Read in truth catalogue---
hdulist=fits.open(folder+'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test_pacs100_50cut.fits')
fcat_sim=hdulist[1].data
hdulist.close()






#error_percent_xidp_250=np.empty((nsources_xidp))
#IQR_xidp_250=np.empty((nsources_xidp))
#accuracy_xidp_250=np.empty((nsources_xidp))
error_percent_xid1_250=np.empty((nsources_xid))
error_percent_xid2_250=np.empty((nsources_xid))
accuracy_xid_250=np.empty((nsources_xid))
IQR_xid_250=np.empty((nsources_xid))

import scipy.stats as stats
#for i in range(0,nsources_xidp):
#    error_percent_xidp_250[i]=stats.percentileofscore(flattened_post[:,i],fcat_sim['S250'][idx_xidp][i])
#    IQR_xidp_250[i]=np.subtract(*np.percentile(flattened_post[:,i],[75.0,25.0]))/np.median(flattened_post[:,i])
#    accuracy_xidp_250[i]=(np.median(flattened_post[:,i])-fcat_sim['S250'][idx_xidp][i])/fcat_sim['S250'][idx_xidp][i]

#-----set up truncated boundaries for DESPHOT----
low_clip=0.0
up_clip=1000.0

for i in range(0,nsources_xid):
    my_mean=fcat['F250'][i]
    my_std=fcat['E250'][i]
    a,b = (low_clip - my_mean)/my_std,(up_clip - my_mean)/my_std
    error_percent_xid1_250[i]=100.0*stats.norm.cdf(fcat_sim['S250'][i],loc=fcat['F250'][i],scale=fcat['E250'][i])
    error_percent_xid2_250[i]=100.0*stats.truncnorm.cdf((fcat_sim['S250'][i]-my_mean)/my_std,a,b)
    lq,uq=stats.truncnorm.interval(0.5,a,b)
    IQR_xid_250[i]=((uq*my_std)-(lq*my_std))/my_mean
    accuracy_xid_250[i]=(my_mean-fcat_sim['S250'][i])/fcat_sim['S250'][i]


bins=np.array([0.0,0.0032,0.135,2.275,15.87,50.0,84.13,97.725,99.865,99.9968,100.0])
ind_3mjy=fcat_sim['S250']>3
hist_xid1,bin_eges_xid1=np.histogram(error_percent_xid1_250[ind_3mjy],bins)
hist_xid2,bin_eges_xid2=np.histogram(error_percent_xid2_250[ind_3mjy],bins)

sigma=np.arange(-4.5,5.0,1)
fig=plt.figure()
plt.plot(sigma,hist_xid1/float(ind_3mjy.sum()), label='DESPHOT1')
plt.plot(sigma,hist_xid2/float(ind_3mjy.sum()), label='DESPHOT2')

#plt.plot(sigma,hist_xidp/float(nsources_xidp), 'r',label='XID+')
plt.xlabel('Flux error density (Sigma)')

plt.legend()
pdf_pages.savefig(fig)

fig2=plt.figure()
#H,xedge,yedge=np.histogram2d(np.log10(fcat_sim['S250'][idx_xidp]),error_percent_xidp_250,bins=[np.arange(-3,3,0.25),bins])
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250']),error_percent_xid2_250,bins=[np.arange(0.5,2.2,0.01),bins])
plt.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],sigma[0],sigma[-1]],origin='low',aspect='auto')
plt.xlabel('S250 mJy')
plt.ylabel('z score')
print H.shape,xedges.shape,yedges.shape
plt.colorbar()
pdf_pages.savefig(fig2)
#bins=[np.arange(-3,3,0.25),bin_eges_xidp]


#flux precision
fig3=plt.figure()
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250']),IQR_xid_250,bins=[np.arange(0.5,2.2,0.01),np.arange(0,2,0.05)])
plt.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto')
plt.xlabel('S250 mJy')
plt.ylabel('IQR mJy/True')
plt.colorbar()
pdf_pages.savefig(fig3)

fig4=plt.figure()
H,xedges,yedges=np.histogram2d(np.log10(fcat_sim['S250']),accuracy_xid_250,bins=[np.arange(0.5,2.2,0.01),np.arange(-2,2,0.01)])
plt.imshow(H.T/np.sum(H.T,axis=0),interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],origin='low',aspect='auto')
plt.xlabel('S250 mJy')
plt.ylabel('Obs -True / True mJy')
plt.colorbar()
pdf_pages.savefig(fig4)

pdf_pages.close()
