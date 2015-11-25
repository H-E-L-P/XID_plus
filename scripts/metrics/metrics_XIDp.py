from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use('PDF')
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from xidplus import XIDp_mod_beta
import pickle
import scipy.stats as stats
from scipy.stats import norm





pdf_pages=PdfPages("error_density_flux_test_uninform.pdf")

from metrics_module import *



    
#---Read in truth catalogue---
folder='/Users/pdh21/HELP/XID_plus_output/sims/lacy/'
#'/research/astro/fir/cclarke/lacey/released/'
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
folder='/Users/pdh21/HELP/XID_plus_output/100micron/old/'
#'/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/old/'
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
ind_3mjy_psw=np.median(flattened_post[:,0:prior250.nsrc],axis=0) >1   
ind_3mjy_pmw=np.median(flattened_post[:,prior250.nsrc+1:(2*prior250.nsrc)+1],axis=0) >1
ind_3mjy_plw=np.median(flattened_post[:,2*prior250.nsrc+2:(3*prior250.nsrc)+2],axis=0) >1
print ind_3mjy_psw.shape
psw_metrics_XIDp=metrics_XIDp(flattened_post[:,np.arange(0,prior250.nsrc,dtype=int)[ind_3mjy_psw]],fcat_sim['S250'][idx_xidp][ind_3mjy_psw])
pmw_metrics_XIDp=metrics_XIDp(flattened_post[:,np.arange(prior250.nsrc+1,(2*prior250.nsrc)+1,dtype=int)[ind_3mjy_pmw]],fcat_sim['S350'][idx_xidp][ind_3mjy_pmw])
plw_metrics_XIDp=metrics_XIDp(flattened_post[:,np.arange(2*prior250.nsrc+2,(3*prior250.nsrc)+2)[ind_3mjy_plw]],fcat_sim['S500'][idx_xidp][ind_3mjy_plw])










bins=np.logspace(0.477, 2.2, num=7)
labels=[r'Z score', r'IQR/$S_{True}$', r'$(S_{Obs}-S_{True})/S_{True}$']
scale=['linear', 'log', 'linear']
ylims=[(-4,4),(1E-2,1E1),(-1,1)]
for i in range(0,3):
    pdf_pages.savefig(metrics_plot(psw_metrics_XIDp[i],fcat_sim['S250'][idx_xidp][ind_3mjy_psw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i]))
    pdf_pages.savefig(metrics_plot(pmw_metrics_XIDp[i],fcat_sim['S350'][idx_xidp][ind_3mjy_pmw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i],cmap=plt.get_cmap('Greens')))
    pdf_pages.savefig(metrics_plot(plw_metrics_XIDp[i],fcat_sim['S500'][idx_xidp][ind_3mjy_plw],bins,[r'$S_{True} (\mathrm{mJy})$',labels[i]],ylims[i],yscale=scale[i],cmap=plt.get_cmap('Reds')))

pdf_pages.close()
