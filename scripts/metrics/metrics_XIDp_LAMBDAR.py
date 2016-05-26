from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use('PDF')
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import xidplus
import pickle
import scipy.stats as stats
from scipy.stats import norm





pdf_pages=PdfPages("error_density_flux_test_uninform_uniform_XIDp_LAMBDAR.pdf")

from metrics_module import *



folder='/research/astro/fir/pdh21/lacey/'
hdulist=fits.open(folder+'lacey_20160525_MillGas.ALLVOLS_cat_PSW_COSMOS_out.fits')
fcat_sim=hdulist[1].data
hdulist.close()

fcat_sim=fcat_sim[(fcat_sim['APPRSO_TOT_EXT'] <19.8) | (fcat_sim['S250']>4*3.87)]

#---match DESPHOT and real catalogues---
#from astropy.coordinates import SkyCoord
#from astropy import units as u
#c= SkyCoord(ra=fcat['INRA']*u.degree,dec=fcat['INDEC']*u.degree)
#c1=SkyCoord(ra=fcat_sim['RA']*u.degree,dec=fcat_sim['DEC']*u.degree)
#idx,d2d,d3d,= c.match_to_catalog_sky(c1)

idx_xidp=(fcat_sim['APPRSO_TOT_EXT'] <19.8) | (fcat_sim['S250']>4*3.87)#cut so that only sources with a 100micron flux of > 50 micro janskys (Roseboom et al. 2010 cut 24 micron sources at 50microJys)
idx_xidpT=(fcat_sim['APPRSO_TOT_EXT'] <19.8) | (fcat_sim['S250']>4*3.87)#cut so that only sources with a 100micron flux of > 50 micro janskys (Roseboom et al. 2010 cut 24 micron sources at 50microJys)


#---Read in XID+ posterior---

#folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'
folder='/lustre/scratch/astro/pdh21/COSMOS_sim/output/'
#'/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/old/'
infile=folder+'Master_prior.pkl'
with open(infile, "rb") as f:
    obj = pickle.load(f)
prior250=obj['psw']
prior350=obj['pmw']    
prior500=obj['plw']

infile=folder+'master_posterior.pkl'

with open(infile, "rb") as f:
    obj = pickle.load(f)
posterior=obj['posterior']

samples,chains,params=posterior.shape

flattened_post=posterior.reshape(samples*chains,params)
nsources_xidp=idx_xidp.size
ind_3mjy_psw=np.median(flattened_post[:,0:prior250.nsrc],axis=0) >1   
ind_3mjy_pmw=np.median(flattened_post[:,prior250.nsrc+1:(2*prior250.nsrc)+1],axis=0) >1
ind_3mjy_plw=np.median(flattened_post[:,2*prior250.nsrc+2:(3*prior250.nsrc)+2],axis=0) >1
print ind_3mjy_psw.shape
psw_metrics_XIDp=metrics_XIDp(flattened_post[:,np.arange(0,prior250.nsrc,dtype=int)[ind_3mjy_psw]],fcat_sim['S250'][idx_xidp][ind_3mjy_psw])
pmw_metrics_XIDp=metrics_XIDp(flattened_post[:,np.arange(prior250.nsrc+1,(2*prior250.nsrc)+1,dtype=int)[ind_3mjy_pmw]],fcat_sim['S350'][idx_xidp][ind_3mjy_pmw])
plw_metrics_XIDp=metrics_XIDp(flattened_post[:,np.arange(2*prior250.nsrc+2,(3*prior250.nsrc)+2)[ind_3mjy_plw]],fcat_sim['S500'][idx_xidp][ind_3mjy_plw])




#-----------------DESPHOT STUFF----------------------------------
folder='/research/astro/fir/HELP/DESPHOT/'
#folder='/Users/pdh21/HELP/XID_plus_output/plot_test/'
hdulist=fits.open(folder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise__PSWXID_S100_50mic_test.fits')
fcat=hdulist[1].data
nsources_xid=fcat.shape[0]
print nsources_xid
hdulist.close()

#---Read in truth catalogue---
hdulist=fits.open(folder+'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test_pacs100_50cut.fits')
fcat_simXID=hdulist[1].data
hdulist.close()

#-----set up truncated boundaries for DESPHOT----
low_clip=0.0
up_clip=1000.0
flattened_post_psw=np.empty((1000,nsources_xid))
flattened_post_pmw=np.empty((1000,nsources_xid))
flattened_post_plw=np.empty((1000,nsources_xid))

from scipy.stats import truncnorm
for i in range(0,nsources_xid):
    my_mean=fcat['F250'][i]
    my_std=fcat['E250'][i]
    a,b = (low_clip - my_mean)/my_std,(up_clip - my_mean)/my_std
    flattened_post_psw[:,i] = (truncnorm.rvs(a, b, size=1000)*my_std)+my_mean
    my_mean=fcat['F350'][i]
    my_std=fcat['E350'][i]
    a,b = (low_clip - my_mean)/my_std,(up_clip - my_mean)/my_std
    flattened_post_pmw[:,i] = (truncnorm.rvs(a, b, size=1000)*my_std)+my_mean
    my_mean=fcat['F500'][i]
    my_std=fcat['E500'][i]
    a,b = (low_clip - my_mean)/my_std,(up_clip - my_mean)/my_std
    flattened_post_plw[:,i] = (truncnorm.rvs(a, b, size=1000)*my_std)+my_mean

ind_1mjy_psw=fcat['F250']> 1
ind_1mjy_pmw=fcat['F350']> 1
ind_1mjy_plw=fcat['F500']> 1

psw_metrics_XID=metrics_XIDp(flattened_post_psw[:,ind_1mjy_psw],fcat_simXID['S250'][ind_1mjy_psw])
pmw_metrics_XID=metrics_XIDp(flattened_post_pmw[:,ind_1mjy_pmw],fcat_simXID['S350'][ind_1mjy_pmw])
plw_metrics_XID=metrics_XIDp(flattened_post_plw[:,ind_1mjy_plw],fcat_simXID['S500'][ind_1mjy_plw])
#-------------------------------------------------------------------------------





bins=np.logspace(0.477, 2.2, num=7)
labels=[r'Z score', r'IQR/$S_{True}$', r'$(S_{Obs}-S_{True})/S_{True}$']
scale=['linear', 'log', 'linear']
ylims=[(-2,3),(1E-2,1E1),(-1,1)]

LAMBDAR_x=np.array([0.029,0.033,0.069,0.150,0.269,1.364])*1000.0
LAMBDAR_y=np.array([0.534,0.336,-0.024,-0.043,-0.0037,0.024])
for i in range(0,3):
    if i < 2:
        pdf_pages.savefig(metrics_plot_nodensity_XIDp(psw_metrics_XIDp[i],fcat_sim['S250'][idx_xidp][ind_3mjy_psw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i]))
    else:
        fig=metrics_plot_nodensity_XIDp(psw_metrics_XIDp[i],fcat_sim['S250'][idx_xidp][ind_3mjy_psw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i])
        ax=fig.get_axes()
        ax[0].plot(LAMBDAR_x,LAMBDAR_y,'ko',linestyle='--')
        pdf_pages.savefig(fig)
    pdf_pages.savefig(metrics_plot_nodensity_XIDp(pmw_metrics_XIDp[i],fcat_sim['S350'][idx_xidp][ind_3mjy_pmw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i],cmap=plt.get_cmap('Greens')))
    pdf_pages.savefig(metrics_plot_nodensity_XIDp(plw_metrics_XIDp[i],fcat_sim['S500'][idx_xidp][ind_3mjy_plw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i],cmap=plt.get_cmap('Reds')))

pdf_pages.close()
