
# coding: utf-8

# In this notebook, I will compare the performance of DESPHOT and XID+ on the Lacey simulated maps mimicking COSMOS. As a prior catalogue I have taken sources with a 100 $\mathrm{\mu m}$ flux greater than $50 \mathrm{\mu Jy}$. A similar flux cut was applied to the 24 micron catalogue to select priors for the original HerMES XID catalogues.

# In order to compare methods, I have run DESPHOT, XID+ and XID+Tiling. I have spent quite a while trying to figure out best mmethod for tiling. Originaly, I had wanted to tile the map up with no overlaps, bit fit each tile with sources that lie within tile and some additional buffer region. I could then use the fact that, "conditional on the fluxes" each tile should be indepdendent. I could then multiply the posteriors from each tile in a similar fashion as a Naive Bayes Classifier. Unfortunetly, multiplying high dimensional posteriors is only feasible if i could model them as Gaussians. After much investigation (i.e. fitting fluxes in log space, changing prior etc), it was obvious I could not get away with this approximation.
# 
# Tiling is now done by creating tiles that overlap in the map. I still fit for sources that extend beyond tile. Now each source has an optimum tile and will lie someway inside that specific tile. I use the posterier information from that optimum tile for that individual source.
# 
# 

# One thing to note. The XID+ run took over a week. The tiling run took ~3 hours. Also, the prior has changed between the two. For XID+, fluxes were fitted in log10 space, but with a uniform prior of -8 to 3 $log_{10} f$. For the tiling, I had changed prior to a normal distribution with $\mu=-1$ and $\sigma=2.2$. This is more like P(D) distribution. Only affect this has had is on sources that it obviously can't fit, the flux value reflects the prior. i.e. for XID+ $log_{10}f=-3$, and for XID+tiling it is more like -1.

# In[1]:

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
#get_ipython().magic(u'matplotlib inline')


# In[2]:

#---Read in DESPHOT catalogue---
folder='/research/astro/fir/HELP/DESPHOT/'
hdulist=fits.open(folder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise__PSWXID_S100_50mic_test.fits')
fcat=hdulist[1].data
ind=fcat['xid'] >= 0
fcat=fcat[ind]
hdulist.close()


# In[3]:

#---Read in truth catalogue---
folder='/research/astro/fir/cclarke/lacey/released/'
hdulist=fits.open(folder+'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test.fits')
fcat_sim=hdulist[1].data
hdulist.close()


# In[4]:

#---Read in XID+ catalogue---
folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/'
hdulist=fits.open(folder+'Tiled_SPIRE_cat_flux_notlog.fits')
fcat_xidp=hdulist[1].data
hdulist.close()


# In[5]:

#---Read in XID+Tiling catalogue---
folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'
hdulist=fits.open(folder+'Tiled_SPIRE_cat_flux_notlog.fits')
fcat_xidpT=hdulist[1].data
hdulist.close()


# In[6]:

#---match DESPHOT and real catalogues---
from astropy.coordinates import SkyCoord
from astropy import units as u
c= SkyCoord(ra=fcat['INRA']*u.degree,dec=fcat['INDEC']*u.degree)
c1=SkyCoord(ra=fcat_sim['RA']*u.degree,dec=fcat_sim['DEC']*u.degree)
idx,d2d,d3d,= c.match_to_catalog_sky(c1)


# In[7]:

#---match XID+ and real catalogues---
c= SkyCoord(ra=fcat_xidp['ra']*u.degree,dec=fcat_xidp['dec']*u.degree)
c1=SkyCoord(ra=fcat_sim['RA']*u.degree,dec=fcat_sim['DEC']*u.degree)
idx_xidp,d2d,d3d,= c.match_to_catalog_sky(c1)


# In[8]:

#---match XID+ and real catalogues---
c= SkyCoord(ra=fcat_xidpT['ra']*u.degree,dec=fcat_xidpT['dec']*u.degree)
c1=SkyCoord(ra=fcat_sim['RA']*u.degree,dec=fcat_sim['DEC']*u.degree)
idx_xidpT,d2d,d3d,= c.match_to_catalog_sky(c1)


# In[9]:

plt.hist(np.log10(fcat_xidpT['flux250_err_u']), bins=np.arange(-8,4,0.5))


# In[10]:

print('asd')


# In[11]:

#--Calculate (Sobs-Strue)/sigma_obs
dis250=(fcat['F250']-fcat_sim['S250'][idx])/fcat['E250']
dis350=(fcat['F350']-fcat_sim['S350'][idx])/fcat['E350']
dis500=(fcat['F500']-fcat_sim['S500'][idx])/fcat['E500']

dis250_xidp=(fcat_xidp['flux250']-fcat_sim['S250'][idx_xidp])/(fcat_xidp['flux250_err_u']-fcat_xidp['flux250'])
dis350_xidp=(fcat_xidp['flux350']-fcat_sim['S350'][idx_xidp])/(fcat_xidp['flux350_err_u']-fcat_xidp['flux350'])
dis500_xidp=(fcat_xidp['flux500']-fcat_sim['S500'][idx_xidp])/(fcat_xidp['flux500_err_u']-fcat_xidp['flux500'])

dis250_xidpT=(fcat_xidpT['flux250']-fcat_sim['S250'][idx_xidpT])/(fcat_xidpT['flux250']-fcat_xidpT['flux250_err_l'])
dis350_xidpT=(fcat_xidpT['flux350']-fcat_sim['S350'][idx_xidpT])/(fcat_xidpT['flux350']-fcat_xidpT['flux350_err_l'])
dis500_xidpT=(fcat_xidpT['flux500']-fcat_sim['S500'][idx_xidpT])/(fcat_xidpT['flux500']-fcat_xidpT['flux500_err_l'])




# In[39]:

plt.figure(figsize=(10,20))
ax=plt.subplot(3,1,1)
plt.plot(fcat_sim['S250'][idx],fcat['F250']+1E-7, 'bo', alpha=0.7,label='DESPHOT')
plt.plot(fcat_sim['S250'][idx_xidp],fcat_xidp['flux250'], 'ro', alpha=0.7, label='XID+')
plt.plot(fcat_sim['S250'][idx_xidpT],fcat_xidpT['flux250'], 'go', alpha=0.7, label='XID+prior')
plt.plot(np.arange(1E-4,1E3),np.arange(1E-4,1E3))
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim((1E-1,1E2))
#plt.ylim((1E-1,1E2))
plt.xlim((1E-4,6E2))
plt.ylim((1E-8,6E2))
ax.set_xticklabels(())
ax.title.set_visible(False)
plt.ylabel('Flux $250_{\mathrm{Output}}$')


ax=plt.subplot(3,1,2)
plt.plot(fcat_sim['S350'][idx],fcat['F350']+1E-7, 'bo', alpha=0.7, label='DESPHOT')
plt.plot(fcat_sim['S350'][idx_xidp],fcat_xidp['flux350'], 'ro', alpha=0.7, label='XID+')
plt.plot(fcat_sim['S350'][idx_xidpT],fcat_xidpT['flux350'], 'go', alpha=0.7, label='XID+prior')
plt.plot(np.arange(1E-4,1E3),np.arange(1E-4,1E3))
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim((1E-1,1E2))
#plt.ylim((1E-1,1E2))
plt.xlim((1E-4,6E2))
plt.ylim((1E-8,6E2))
ax.set_xticklabels(())
ax.title.set_visible(False)
plt.ylabel('Flux $350_{\mathrm{Output}}$')
ax=plt.subplot(3,1,3)
plt.plot(fcat_sim['S500'][idx],fcat['F500']+1E-7, 'bo', alpha=0.7, label='DESPHOT')
plt.plot(fcat_sim['S500'][idx_xidp],fcat_xidp['flux500'], 'ro', alpha=0.7, label='XID+')
plt.plot(fcat_sim['S500'][idx_xidpT],fcat_xidpT['flux500'], 'go', alpha=0.7, label='XID+prior')
plt.plot(np.arange(1E-4,1E3),np.arange(1E-4,1E3))
#plt.xscale('log')
#plt.yscale('log')
plt.xlim((1E-4,6E2))
plt.ylim((1E-8,6E2))
plt.ylabel('Flux $500_{\mathrm{Output}}$')
plt.subplots_adjust(hspace=0.0)



plt.xlabel('Flux$_{\mathrm{Truth}}$')
plt.legend(loc=2)
plt.savefig("Out_vs_true_lin.pdf")



## Flux Accuracy

# In[35]:

nbins=5

f_acc250=fcat['F250']-fcat_sim['S250'][idx]
f_acc250_xidp=fcat_xidp['flux250']-fcat_sim['S250'][idx_xidp]
f_acc250_xidpT=fcat_xidpT['flux250']-fcat_sim['S250'][idx_xidpT]
f_acc_bins=np.logspace(0,2.3,nbins)
ind_bins250=np.digitize(fcat_sim['S250'][idx],f_acc_bins)
ind_bins250_xidp=np.digitize(fcat_sim['S250'][idx_xidp],f_acc_bins)
ind_bins250_xidpT=np.digitize(fcat_sim['S250'][idx_xidpT],f_acc_bins)

f_acc350=fcat['F350']-fcat_sim['S350'][idx]
f_acc350_xidp=fcat_xidp['flux350']-fcat_sim['S350'][idx_xidp]
f_acc350_xidpT=fcat_xidpT['flux350']-fcat_sim['S350'][idx_xidpT]
#f_acc_bins=np.logspace(0,2,nbins)
ind_bins350=np.digitize(fcat_sim['S350'][idx],f_acc_bins)
ind_bins350_xidp=np.digitize(fcat_sim['S350'][idx_xidp],f_acc_bins)
ind_bins350_xidpT=np.digitize(fcat_sim['S350'][idx_xidpT],f_acc_bins)

f_acc500=fcat['F500']-fcat_sim['S500'][idx]
f_acc500_xidp=fcat_xidp['flux500']-fcat_sim['S500'][idx_xidp]
f_acc500_xidpT=fcat_xidpT['flux500']-fcat_sim['S500'][idx_xidpT]
#f_acc_bins=np.logspace(0,2,nbins)
ind_bins500=np.digitize(fcat_sim['S500'][idx],f_acc_bins)
ind_bins500_xidp=np.digitize(fcat_sim['S500'][idx_xidp],f_acc_bins)
ind_bins500_xidpT=np.digitize(fcat_sim['S500'][idx_xidpT],f_acc_bins)

print np.mean(f_acc250),np.mean(f_acc250_xidp),np.mean(f_acc250_xidpT)



f_acc_250=np.empty((5,nbins))
f_acc_250_xidp=np.empty((5,nbins))
f_acc_250_xidpT=np.empty((5,nbins))

f_acc_350=np.empty((5,nbins))
f_acc_350_xidp=np.empty((5,nbins))
f_acc_350_xidpT=np.empty((5,nbins))

f_acc_500=np.empty((5,nbins))
f_acc_500_xidp=np.empty((5,nbins))
f_acc_500_xidpT=np.empty((5,nbins))

for i in range(0,nbins):
    f_acc_250[:,i]=np.percentile(f_acc250[ind_bins250 == i],[0.13,15.87,50,84.13,99.87])
    f_acc_250_xidp[:,i]=np.percentile(f_acc250_xidp[ind_bins250_xidp == i],[0.13,15.87,50,84.13,99.87])
    f_acc_250_xidpT[:,i]=np.percentile(f_acc250_xidpT[ind_bins250_xidpT == i],[0.13,15.87,50,84.13,99.87])
    
    f_acc_350[:,i]=np.percentile(f_acc350[ind_bins350 == i],[0.13,15.87,50,84.13,99.87])
    f_acc_350_xidp[:,i]=np.percentile(f_acc350_xidp[ind_bins350_xidp == i],[0.13,15.87,50,84.13,99.87])
    f_acc_350_xidpT[:,i]=np.percentile(f_acc350_xidpT[ind_bins350_xidpT == i],[0.13,15.87,50,84.13,99.87])
    
    
    f_acc_500[:,i]=np.percentile(f_acc500[ind_bins500 == i],[0.13,15.87,50,84.13,99.87])
    f_acc_500_xidp[:,i]=np.percentile(f_acc500_xidp[ind_bins500_xidp == i],[0.13,15.87,50,84.13,99.87])
    f_acc_500_xidpT[:,i]=np.percentile(f_acc500_xidpT[ind_bins500_xidpT == i],[0.13,15.87,50,84.13,99.87])
    
    
plt.figure(figsize=(10,20))
ax=plt.subplot(3,1,1)
plt.plot(f_acc_bins,f_acc_250[2,:],label='DESPHOT')
plt.fill_between(f_acc_bins,f_acc_250[0,:],f_acc_250[4,:],facecolor='blue',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_250[1,:],f_acc_250[3,:],facecolor='blue',alpha=0.3)

plt.plot(f_acc_bins,f_acc_250_xidp[2,:],'g',label='XID+')
plt.fill_between(f_acc_bins,f_acc_250_xidp[0,:],f_acc_250_xidp[4,:],facecolor='red',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_250_xidp[1,:],f_acc_250_xidp[3,:],facecolor='red',alpha=0.3)

plt.plot(f_acc_bins,f_acc_250_xidpT[2,:],'r',label='XID+prior')
plt.fill_between(f_acc_bins,f_acc_250_xidpT[0,:],f_acc_250_xidpT[4,:],facecolor='green',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_250_xidpT[1,:],f_acc_250_xidpT[3,:],facecolor='green',alpha=0.3)
ax.set_xticklabels(())
ax.title.set_visible(False)

ax=plt.subplot(3,1,2)
plt.plot(f_acc_bins,f_acc_350[2,:],label='DESPHOT')
plt.fill_between(f_acc_bins,f_acc_350[0,:],f_acc_350[4,:],facecolor='blue',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_350[1,:],f_acc_350[3,:],facecolor='blue',alpha=0.3)

plt.plot(f_acc_bins,f_acc_350_xidp[2,:],'g',label='XID+')
plt.fill_between(f_acc_bins,f_acc_350_xidp[0,:],f_acc_350_xidp[4,:],facecolor='red',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_350_xidp[1,:],f_acc_350_xidp[3,:],facecolor='red',alpha=0.3)

plt.plot(f_acc_bins,f_acc_350_xidpT[2,:],'r',label='XID+prior')
plt.fill_between(f_acc_bins,f_acc_350_xidpT[0,:],f_acc_350_xidpT[4,:],facecolor='green',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_350_xidpT[1,:],f_acc_350_xidpT[3,:],facecolor='green',alpha=0.3)
plt.ylabel('$S_{\mathrm{Output}}$ - $S_{\mathrm{Truth}} \mathrm{(mJy)}$')

ax.set_xticklabels(())
ax.title.set_visible(False)

ax=plt.subplot(3,1,3)
plt.plot(f_acc_bins,f_acc_500[2,:],label='DESPHOT')
plt.fill_between(f_acc_bins,f_acc_500[0,:],f_acc_500[4,:],facecolor='blue',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_500[1,:],f_acc_500[3,:],facecolor='blue',alpha=0.3)

plt.plot(f_acc_bins,f_acc_500_xidp[2,:],'g',label='XID+')
plt.fill_between(f_acc_bins,f_acc_500_xidp[0,:],f_acc_500_xidp[4,:],facecolor='red',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_500_xidp[1,:],f_acc_500_xidp[3,:],facecolor='red',alpha=0.3)

plt.plot(f_acc_bins,f_acc_500_xidpT[2,:],'r',label='XID+prior')
plt.fill_between(f_acc_bins,f_acc_500_xidpT[0,:],f_acc_500_xidpT[4,:],facecolor='green',alpha=0.1)
plt.fill_between(f_acc_bins,f_acc_500_xidpT[1,:],f_acc_500_xidpT[3,:],facecolor='green',alpha=0.3)
plt.subplots_adjust(hspace=0.0)

plt.legend()
#plt.ylabel('Flux $250_{\mathrm{Output}}$ - Flux $250_{\mathrm{Truth}} \mathrm{(mJy)}$')
plt.xlabel('$S_{\mathrm{Truth}} \mathrm{(mJy)}$')
plt.savefig("flux_accuracy_binned.pdf")


# In[41]:

plt.figure(figsize=(10,20))
ax=plt.subplot(3,1,1)
plt.hist(f_acc250, bins=np.arange(-15,15,0.5), alpha=0.5,normed=True)
plt.hist(f_acc250_xidp, bins=np.arange(-15,15,0.5),alpha=0.5, color='r',normed=True)
plt.hist(f_acc250_xidpT, bins=np.arange(-15,15,0.5),alpha=0.5, color='g',normed=True)

ax.set_xticklabels(())
ax.title.set_visible(False)

ax=plt.subplot(3,1,2)
plt.hist(f_acc350, bins=np.arange(-15,15,0.5), alpha=0.5,normed=True)
plt.hist(f_acc350_xidp, bins=np.arange(-15,15,0.5),alpha=0.5, color='r',normed=True)
plt.hist(f_acc350_xidpT, bins=np.arange(-15,15,0.5),alpha=0.5, color='g',normed=True)

ax.set_xticklabels(())
ax.title.set_visible(False)

ax=plt.subplot(3,1,3)
plt.hist(f_acc500, bins=np.arange(-15,15,0.5), alpha=0.5,normed=True, label='DESPHOT')
plt.hist(f_acc500_xidp, bins=np.arange(-15,15,0.5),alpha=0.5, color='r',normed=True,label='XID+')
plt.hist(f_acc500_xidpT, bins=np.arange(-15,15,0.5),alpha=0.5, color='g',normed=True)

plt.subplots_adjust(hspace=0.0)

plt.legend()
plt.xlabel('$S_{\mathrm{Output}}$ - $S_{\mathrm{Truth}} \mathrm{(mJy)}$')
plt.savefig("flux_accuracy_hist.pdf")


##  Completeness

# In[19]:

SNR250=fcat['F250']/fcat['E250']
SNR250=fcat['F250']/fcat['E250']
SNR250=fcat['F250']/fcat['E250']
def Completeness(output_flux,output_flux_error,input_flux,match,bins,sigma):
    SN=output_flux/output_flux_error
    #ind=SN>sigma

    print ind.sum(),SN.size
    #output_flux[ind]
    
    ind_bins_input=np.digitize(input_flux,bins)
    ind_bins_output=np.digitize(output_flux,bins)
    ind_TP=(ind_bins_input[match] == ind_bins_output) & (SN>sigma)
    complete=np.empty((bins.size))
    
    for i in range(0,bins.size):
        num=np.float(sum(ind_bins_output[ind_TP] == i))
        denom=sum(ind_bins_input == i)
        if denom !=0:
            complete[i]=num/denom
        if denom ==0:
            complete[i]=2
    return complete


# In[20]:

bins_comp=np.logspace(0,2.5,num=6)
comp250=Completeness(fcat['F250'],fcat['E250'],fcat_sim['S250'],idx,bins_comp,3)
comp350=Completeness(fcat['F350'],fcat['E350'],fcat_sim['S350'],idx,bins_comp,3)
comp500=Completeness(fcat['F500'],fcat['E500'],fcat_sim['S500'],idx,bins_comp,3)
comp250_xidp=Completeness(fcat_xidp['flux250'],fcat_xidp['flux250']-fcat_xidp['flux250_err_l'],fcat_sim['S250'],idx_xidp,bins_comp,3)
comp350_xidp=Completeness(fcat_xidp['flux350'],fcat_xidp['flux350']-fcat_xidp['flux350_err_l'],fcat_sim['S350'],idx_xidp,bins_comp,3)
comp500_xidp=Completeness(fcat_xidp['flux500'],fcat_xidp['flux500']-fcat_xidp['flux500_err_l'],fcat_sim['S500'],idx_xidp,bins_comp,3)
comp250_xidpT=Completeness(fcat_xidpT['flux250'],fcat_xidpT['flux250']-fcat_xidpT['flux250_err_l'],fcat_sim['S250'],idx_xidp,bins_comp,3)
comp350_xidpT=Completeness(fcat_xidpT['flux350'],fcat_xidpT['flux350']-fcat_xidpT['flux350_err_l'],fcat_sim['S350'],idx_xidp,bins_comp,3)
comp500_xidpT=Completeness(fcat_xidpT['flux500'],fcat_xidpT['flux500']-fcat_xidpT['flux500_err_l'],fcat_sim['S500'],idx_xidp,bins_comp,3)


# In[21]:

plt.figure(figsize=(10,20))
ax=plt.subplot(3,1,1)
plt.plot(bins_comp,comp250,label='DESPHOT')
plt.plot(bins_comp,comp250_xidp,label='XID+')
plt.plot(bins_comp,comp250_xidpT,label='XID+Tiling')
ax.set_xticklabels(())
ax.set_xscale('log')
ax.title.set_visible(False)
ax=plt.subplot(3,1,2)
plt.plot(bins_comp,comp350,label='DESPHOT')
plt.plot(bins_comp,comp350_xidp,label='XID+')
plt.plot(bins_comp,comp350_xidpT,label='XID+Tiling')
ax.set_xticklabels(())
ax.set_xscale('log')

ax.title.set_visible(False)
ax=plt.subplot(3,1,3)
plt.plot(bins_comp,comp500,label='DESPHOT')
plt.plot(bins_comp,comp500_xidp,label='XID+')
plt.plot(bins_comp,comp500_xidpT,label='XID+Tiling')
ax.set_xscale('log')

plt.legend()
plt.xlabel('$S_{\mathrm{Truth}} \mathrm{(mJy)}$')


# In[22]:

plt.figure(figsize=(20,20))
plt.plot(fcat_sim['S350'][idx],fcat['F350'], 'bo', label='DESPHOT')
plt.plot(fcat_sim['S350'][idx_xidp],fcat_xidp['flux350'], 'ro', label='XID+')
plt.plot(fcat_sim['S350'][idx_xidpT],fcat_xidpT['flux350'], 'go', label='XID+Tiling')

plt.xlabel('Flux $350_{\mathrm{Truth}}$')
plt.ylabel('Flux $350_{\mathrm{Output}}$')
plt.legend()


# In[21]:

plt.figure(figsize=(20,20))
plt.plot(fcat_sim['S500'][idx],fcat['F500'], 'bo', label='DESPHOT')
plt.plot(fcat_sim['S500'][idx_xidp],fcat_xidp['flux500'], 'ro', label='XID+')
plt.plot(fcat_sim['S500'][idx_xidpT],fcat_xidpT['flux500'], 'go', label='XID+Tiling')

plt.xlabel('Flux $500_{\mathrm{Truth}}$')
plt.ylabel('Flux $500_{\mathrm{Output}}$')
plt.legend()


# Note, DESPHOT, XID+ and XID+Tiling give basically the same results for sources with a high flux. The figure below is the same but in log space (just for the 250). Difference between XID+ and XID+Tiling is due to difference in prior and only affects faint sources.

# In[34]:

plt.figure(figsize=(20,20))
plt.loglog(fcat_sim['S250'][idx],fcat['F250']+1E-6, 'bo', label='DESPHOT',alpha=0.3)
plt.loglog(fcat_sim['S250'][idx_xidp],fcat_xidp['flux250'], 'ro', label='XID+',alpha=0.3)
plt.loglog(fcat_sim['S250'][idx_xidpT],fcat_xidpT['flux250'], 'go', label='XID+prior',alpha=0.3)

plt.xlabel('Flux $250_{\mathrm{Truth}}$')
plt.ylabel('Flux $250_{\mathrm{Output}}$')
plt.legend(loc=3)


## Flux density error

# In[23]:

bins=np.arange(-20,20,0.2)
fig=plt.figure(figsize=(10,20))
#---250---
ax=plt.subplot(3,1,1)
plt.hist(dis250,bins=bins,alpha=0.4)
plt.hist(dis250_xidp,bins=bins,alpha=0.4)
plt.hist(dis250_xidpT,bins=bins,alpha=0.4)
plt.ylabel('Counts $250\mu m$')
ax.set_xticklabels(())
ax.title.set_visible(False)
#---350---
ax=plt.subplot(3,1,2)
plt.hist(dis350,bins=bins,alpha=0.4)
plt.hist(dis350_xidp,bins=bins,alpha=0.4)
plt.hist(dis350_xidpT,bins=bins,alpha=0.4)
plt.ylabel('Counts $350\mu m$')

ax.set_xticklabels(())
ax.title.set_visible(False)
#---500---
ax=plt.subplot(3,1,3)
plt.hist(dis500,bins=bins,alpha=0.4,label='DESPHOT')
plt.hist(dis500_xidp,bins=bins,alpha=0.4, label='XID+')
plt.hist(dis500_xidpT,bins=bins,alpha=0.4,label='XID+Tiling')
plt.ylabel('Counts $500\mu m$')

plt.subplots_adjust(hspace=0.0)
plt.xlabel('$(S_{obs}-S_{true})/\sigma_{obs}$')
plt.legend()


# In[24]:

plt.hist(dis250_xidpT,bins=bins,alpha=0.4)


# As in Roseboom et al. 2010. I have calculated the distribution of observed flux density error, normalised by $\sigma$. XID+ and XID+ tiling has a peak which is 0 suggesting both XID+ methods are better at approximating flux density error. NOTE: at the moment. My catalogues are giving marginalised 16th, 50th and 84th percentiles. The above plots have been made assuming $\sigma=84th-50th$ percentile. 

# In[24]:




## Flux distribution

# Read in the full posterior for XID+, to compare flux distribution.

# In[19]:

#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/'

infile=output_folder+'Lacey_rbandcut_19_8_log_flux.pkl'
with open(infile, "rb") as f:
    obj = pickle.load(f)
prior250=obj['psw']
prior350=obj['pmw']    
prior500=obj['plw']

posterior=obj['posterior']


# Read in full posterior for combined tiles

# In[16]:

#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/Tiling/log_uniform_prior/'

infile=output_folder+'Tiled_master_Lacey_rbandcut_19_8_notlog_flux.pkl'
with open(infile, "rb") as f:
    obj_pT = pickle.load(f)
prior250_pT=obj_pT['psw']
prior350_pT=obj_pT['pmw']    
prior500_pT=obj_pT['plw']

posterior_pT=obj_pT['posterior']


# In[17]:

samples,chains,params=posterior.stan_fit.shape
flattened_post=posterior.stan_fit.reshape(samples*chains,params)


# In[18]:

samples_pT,chains_pT,params_pT=posterior_pT.stan_fit.shape
flattened_post_pT=np.log10(posterior_pT.stan_fit.reshape(samples_pT*chains_pT,params_pT))


# In[19]:

bins=np.arange(-6,4,0.1)
hist_sim,bin_edges=np.histogram(np.log10(fcat_sim['S250'][idx_xidp]),bins=bins)


# In[20]:

fig=plt.figure(figsize=(20,20))
hist_post=np.empty((hist_sim.size,flattened_post.shape[0]))
hist_post_pT=np.empty((hist_sim.size,flattened_post_pT.shape[0]))

print bins.size,hist_sim.size
plt.plot(bins[1:],hist_sim, label='Truth')
for i in np.arange(0,flattened_post.shape[0]):
    hist_samp,bin_edges=np.histogram(flattened_post[i,0:prior250.nsrc],bins=bins)
    hist_post[:,i]=hist_samp
    hist_samp_pT,bin_edges_pT=np.histogram(flattened_post_pT[i,0:prior250_pT.nsrc],bins=bins)
    hist_post_pT[:,i]=hist_samp_pT    
    #plt.plot(bins[1:],hist_post,'g',alpha=0.5)
plt.plot(bins[1:],np.mean(hist_post,axis=1),'g',label='XID+ mean')
plt.fill_between(bins[1:],np.percentile(hist_post,84,axis=1),np.percentile(hist_post,16,axis=1),facecolor='green',alpha=0.2,label='XID+ 84th-16th percentile')
plt.plot(bins[1:],np.mean(hist_post_pT,axis=1),'r',label='XID+Tiling mean')
plt.fill_between(bins[1:],np.percentile(hist_post_pT,84,axis=1),np.percentile(hist_post_pT,16,axis=1),facecolor='red',alpha=0.2,label='XID+Tiling 84th-16th percentile')
plt.xlabel('$\log_{10}S_{250\mathrm{\mu m}} (\mathrm{mJy})$')
plt.legend()


# In order to check the models, I should use posterior predictive checking. 

# In[26]:

def yrep_map(prior,fvec):
    from scipy.sparse import coo_matrix
    

    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    f=coo_matrix((fvec, (range(0,prior.nsrc+1),np.zeros(prior.nsrc+1))), shape=(prior.nsrc+1, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc+1))
    rmap_temp=(A*f)
    pred_map=np.empty_like(prior.im)
    
    pred_map[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)+np.random.randn(prior.snpix)*prior.snim
    
    return pred_map

def map_model(prior,fvec):
    from scipy.sparse import coo_matrix
    

    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    f=coo_matrix((fvec, (range(0,prior.nsrc+1),np.zeros(prior.nsrc+1))), shape=(prior.nsrc+1, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc+1))
    rmap_temp=(A*f)
    im_model=np.empty_like(prior.im)
    
    im_model[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)#+np.random.randn(prior.snpix)*prior.snim]
    return im_model

def map_orig(prior):
    from scipy.sparse import coo_matrix
    
    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    im_orig=np.empty_like(prior.im)
    im_orig=prior.im
    return im_orig
def map_orig_nim(prior):
    from scipy.sparse import coo_matrix
    
    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    f=coo_matrix((fvec, (range(0,prior.nsrc+1),np.zeros(prior.nsrc+1))), shape=(prior.nsrc+1, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc+1))
    rmap_temp=(A*f)
    im_orig_nim=np.empty_like(prior.im)
    
    im_orig_nim=prior.nim
    return im_orig_nim


# In[55]:

def make_yrep_maps(prior,posterior,N=16):
    samp,chains,params=posterior.shape
    flattened_posterior=posterior.reshape(samp*chains,params)
    yrep_inds=np.round(np.random.random((N))*samp*chains)
    pred_maps=np.empty((N,prior.im.shape[0],prior.im.shape[1]))

    for i in range(0,N):
        pred_maps[i,:,:]=yrep_map(prior,flattened_posterior[yrep_inds[i],:])
    return pred_maps

def make_yrep_model_maps(prior,posterior,N=16):
    samp,chains,params=posterior.shape
    flattened_posterior=posterior.reshape(samp*chains,params)
    yrep_inds=np.round(np.random.random((N))*samp*chains)
    pred_maps=np.empty((N,prior.im.shape[0],prior.im.shape[1]))

    for i in range(0,N):
        pred_maps[i,:,:]=map_model(prior,flattened_posterior[yrep_inds[i],:])
    return pred_maps
    
    
    
    
def pixel_histogram_check(prior,pred_maps,N=16):
    """function that creates a plot comparing the pixel histogram from N $\mathrm{y}^{rep}$ \n 
    repititons from the posterior, with that from the map"""
    nx_plot=np.sqrt(N)
    ny_plot=np.ceil(N/np.round(np.sqrt(N)))
    fig=plt.figure(figsize=(20,20))
    im_orig=map_orig(prior)
    bins=np.arange(-20,50)
    sy_min=np.min(prior.sy_pix)
    sx_min=np.min(prior.sx_pix)
    sy_max=np.max(prior.sy_pix)
    sx_max=np.max(prior.sx_pix)
    sx_ind=(prior.sim != 0.0) & (prior.sim != 1.0)#(prior.sx_pix>sx_min+20) & (prior.sx_pix <sx_max-20)
    sy_ind=(prior.sim != 0.0) & (prior.sim != 1.0)#(prior.sy_pix>sy_min+20) & (prior.sy_pix <sy_max-20)

    
    for i in range(0,N):
        ax=fig.add_subplot(nx_plot,ny_plot,i)
        ax.hist(pred_maps[i,prior.sy_pix[sy_ind],prior.sx_pix[sx_ind]],alpha=0.5,color='blue',bins=bins)
        ax.hist(im_orig[prior.sy_pix[sy_ind],prior.sx_pix[sx_ind]],alpha=0.5,color='green',bins=bins)
        ax.set_xlim(-20,50)
        
def map_check(prior,pred_maps,N=16):
    nx_plot=np.sqrt(N)
    ny_plot=np.ceil(N/np.round(np.sqrt(N)))
    sy_min=np.min(prior.sy_pix)
    sx_min=np.min(prior.sx_pix)
    sy_max=np.max(prior.sy_pix)
    sx_max=np.max(prior.sx_pix)
    fig_orig=plt.figure(figsize=(10,10))
    ax_orig=fig_orig.add_subplot(1,1,1)
    ax_orig.imshow(prior.im,interpolation='nearest',vmin=-10,vmax=50)
    ax_orig.set_xlim(sx_min+20,sx_max-20)
    ax_orig.set_ylim(sy_min+20,sy_max-20)
    fig=plt.figure(figsize=(20,20))
    im_orig=map_orig(prior)

    for i in range(0,N):
        ax=fig.add_subplot(nx_plot,ny_plot,i)
        ax.imshow(pred_maps[i,:,:],interpolation='nearest',vmin=-10,vmax=50)
        ax.set_xlim(sx_min+20,sx_max-20)
        ax.set_ylim(sy_min+20,sy_max-20)

        
def res_map_check(prior,pred_maps,N=16):
    nx_plot=np.sqrt(N)
    ny_plot=np.ceil(N/np.round(np.sqrt(N)))
    sy_min=np.min(prior.sy_pix)
    sx_min=np.min(prior.sx_pix)
    sy_max=np.max(prior.sy_pix)
    sx_max=np.max(prior.sx_pix)
    #fig_orig=plt.figure(figsize=(10,10))
    #ax_orig=fig_orig.add_subplot(1,1,1)
    #ax_orig.imshow(prior.im,interpolation='nearest',vmin=-10,vmax=50)
    #ax_orig.set_xlim(sx_min-10,sx_max+10)
    #ax_orig.set_ylim(sy_min-10,sy_max+10)
    fig=plt.figure(figsize=(20,20))
    im_orig=map_orig(prior)

    for i in range(0,N):
        ax=fig.add_subplot(nx_plot,ny_plot,i)
        ax.imshow(prior.im-pred_maps[i,:,:],interpolation='nearest',vmin=-10,vmax=10)
        ax.set_xlim(sx_min+20,sx_max-20)
        ax.set_ylim(sy_min+20,sy_max-20)
        
def res_map_hist(prior,pred_maps,N=16):
    nx_plot=np.sqrt(N)
    ny_plot=np.ceil(N/np.round(np.sqrt(N)))
    sy_min=np.min(prior.sy_pix)
    sx_min=np.min(prior.sx_pix)
    sy_max=np.max(prior.sy_pix)
    sx_max=np.max(prior.sx_pix)
    sx_ind=(prior.sx_pix>sx_min+20) & (prior.sx_pix <sx_max-20)
    sy_ind=(prior.sy_pix>sy_min+20) & (prior.sy_pix <sy_max-20)
    #fig_orig=plt.figure(figsize=(10,10))
    #ax_orig=fig_orig.add_subplot(1,1,1)
    #ax_orig.imshow(prior.im,interpolation='nearest',vmin=-10,vmax=50)
    #ax_orig.set_xlim(sx_min-10,sx_max+10)
    #ax_orig.set_ylim(sy_min-10,sy_max+10)
    fig=plt.figure(figsize=(20,20))
    im_orig=map_orig(prior)
    bins=np.arange(-10,10,0.1)
    for i in range(0,N):
        ax=fig.add_subplot(nx_plot,ny_plot,i)
        ax.hist(im_orig[prior.sy_pix[sy_ind],prior.sx_pix[sx_ind]]-pred_maps[i,prior.sy_pix[sy_ind],prior.sx_pix[sx_ind]],bins=bins)
        #ax.set_xlim(sx_min+20,sx_max-20)
        #ax.set_ylim(sy_min+20,sy_max-20) 
    
    
def res_map_check_sum(prior,pred_maps,N=16): 
    nx_plot=np.sqrt(N)
    ny_plot=np.ceil(N/np.round(np.sqrt(N)))
    sy_min=np.min(prior.sy_pix)
    sx_min=np.min(prior.sx_pix)
    sy_max=np.max(prior.sy_pix)
    sx_max=np.max(prior.sx_pix)
    #fig_orig=plt.figure(figsize=(10,10))
    #ax_orig=fig_orig.add_subplot(1,1,1)
    #ax_orig.imshow(prior.im,interpolation='nearest',vmin=-10,vmax=50)
    #ax_orig.set_xlim(sx_min-10,sx_max+10)
    #ax_orig.set_ylim(sy_min-10,sy_max+10)
    fig=plt.figure(figsize=(20,20))
    im_orig=map_orig(prior)
    res_sum=np.empty_like(prior.im)
    for i in range(0,N):
        res_sum+=prior.im-pred_maps[i,:,:]
    fig_orig=plt.figure(figsize=(10,10))
    ax_orig=fig_orig.add_subplot(1,1,1)
    ax_orig.imshow(res_sum/N,interpolation='nearest',vmin=-10,vmax=10)
    ax_orig.set_xlim(sx_min+20,sx_max-20)
    ax_orig.set_ylim(sy_min+20,sy_max-20)


# In[33]:

chains=np.power(10.0,obj['posterior'].stan_fit)
background_inds=[obj['psw'].nsrc,2*obj['psw'].nsrc+1,3*obj['psw'].nsrc+2]
chains[:,:,background_inds]=np.log10(chains[:,:,background_inds])
pred_maps=make_yrep_maps(obj['psw'],chains[:,:,0:obj['psw'].nsrc+1],N=16)
pred_mod_maps=make_yrep_model_maps(obj['psw'],chains[:,:,0:obj['psw'].nsrc+1],N=16)


# In[34]:

map_check(obj['psw'],pred_mod_maps,N=16)


# In[35]:

map_check(obj['psw'],pred_maps,N=16)


# In[36]:

res_map_check(obj['psw'],pred_mod_maps,N=16)


# In[40]:

chains_pT=obj_pT['posterior'].stan_fit
pred_maps_pT=make_yrep_maps(obj['psw'],chains_pT[:,:,0:obj_pT['psw'].nsrc+1],N=16)
pred_mod_maps_pT=make_yrep_model_maps(obj['psw'],chains_pT[:,:,0:obj_pT['psw'].nsrc+1],N=16)


# In[41]:

res_map_check(obj['psw'],pred_mod_maps,N=16)


# In[56]:

pixel_histogram_check(obj['psw'],pred_maps,N=16)


# In[57]:

pixel_histogram_check(obj['psw'],pred_maps_pT,N=16)


# In[ ]:



