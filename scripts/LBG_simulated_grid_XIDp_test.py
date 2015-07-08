import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import XIDp_mod_beta as xid_mod
import os
import sys

#Folder containing maps
imfolder='/research/astro/fir/HELP/Simple_grid_sim/'
#field
field='COSMOS'
#SMAP version
SMAPv='4.2'


# In[7]:
mu_sim=sys.argv[1]
sig_sim=sys.argv[2]
pswfits=imfolder+'COSMOS_psw_'+str(mu_sim)+'_'+str(sig_sim)+'.fits'#SPIRE 250 map
#pmwfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PMW_hipe.fits.gz'#SPIRE 350 map
#plwfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PLW_hipe.fits.gz'#SPIRE 500 map


#----output folder-----------------
#output_folder='/research/astro/fir/HELP/XID_plus_output/Tiling/log_uniform_prior_test/'
output_folder=imfolder
# In[8]:

#Folder containing prior input catalogue
folder=imfolder
#prior catalogue
prior_cat='COSMOS_psw_prior_cat'+str(mu_sim)+'_'+str(sig_sim)+'.fits'

hdulist = fits.open(folder+prior_cat)
fcat=hdulist[1].data
hdulist.close()
inra=fcat['RA']
indec=fcat['Dec']
f_src=fcat['Flux']#apparent r band mag
df_src=f_src
nrealcat=fcat.size
bkg250=-3.0#fcat['bkg250'][0]
bkg350=0#fcat['bkg350'][0]
bkg500=0#fcat['bkg500'][0]
print inra
ra_min=np.min(inra)
ra_max=np.max(inra)
dec_min=np.min(indec)
dec_max=np.max(indec)
tile=np.array([[ra_min,dec_min],[ra_max,dec_min],[ra_max,dec_max],[ra_min,dec_max]]).T






# Open images and noise maps and use WCS module in astropy to get header information

# In[9]:

#-----250-------------
hdulist = fits.open(pswfits)
im250phdu=hdulist[0].header
im250hdu=hdulist[1].header

im250=hdulist[1].data*1.0E3
nim250=hdulist[2].data*1.0E3
w_250 = wcs.WCS(hdulist[1].header)
pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----350-------------
#hdulist = fits.open(pmwfits)
#im350phdu=hdulist[0].header
#im350hdu=hdulist[1].header

#im350=hdulist[1].data*1.0E3
#nim350=hdulist[2].data*1.0E3
#w_350 = wcs.WCS(hdulist[1].header)
#pixsize350=3600.0*w_350.wcs.cd[1,1] #pixel size (in arcseconds)
#hdulist.close()
#-----500-------------
#hdulist = fits.open(plwfits)
#im500phdu=hdulist[0].header
#im500hdu=hdulist[1].header
#im500=hdulist[1].data*1.0E3
#nim500=hdulist[2].data*1.0E3
#w_500 = wcs.WCS(hdulist[1].header)
#pixsize500=3600.0*w_500.wcs.cd[1,1] #pixel size (in arcseconds)
#hdulist.close()


# Since I am only testing, redo this so that I only fit sources within a given range of the mean ra and dec position of the prior list

# In[12]:

##define range
#ra_mean=np.mean(inra)
#dec_mean=np.mean(indec)
#p_range=0.1
##check if sources are within range and if the nearest pixel has a finite value 

#sgood=(inra > ra_mean-p_range) & (inra < ra_mean+p_range) & (indec > dec_mean-p_range) & (indec < dec_mean+p_range)

#--------flux cut on simulation----
##

#--------SEGMENTATION--------------------
#how many tiles are there?
# Point response information, at the moment its 2D Gaussian,

#pixsize array (size of pixels in arcseconds)
pixsize=np.array([pixsize250])
#point response function for the three bands
prfsize=np.array([18.15])
#use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
from astropy.convolution import Gaussian2DKernel


#Set prior classes
#---prior250--------
prior250=xid_mod.prior(im250,nim250,im250phdu,im250hdu)#Initialise with map, uncertianty map, wcs info and primary header
prior250.set_tile(tile,0.01)#Set tile, using a buffer size of 0.01 deg (36'' which is fwhm of PLW)
prior250.prior_cat_stack(inra,indec,prior_cat)#Set input catalogue
prior250.prior_bkg(bkg250,5)#Set prior on background




#thdulist,prior250,prior350,prior500,posterior=xid_mod.fit_SPIRE(prior250,prior350,prior500)

#-----------fit using real beam--------------------------
#PSF_250,px_250,py_250=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PSW_bgmod9_1arcsec.fits',pixsize[0])
#PSF_350,px_350,py_350=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PMW_bgmod9_1arcsec.fits',pixsize[1])
#PSF_500,px_500,py_500=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PLW_bgmod9_1arcsec.fits',pixsize[2])
##---------fit using Gaussian beam-----------------------
prf250=Gaussian2DKernel(prfsize[0]/2.355,x_size=101,y_size=101)
prf250.normalize(mode='peak')


pind250=np.arange(0,101,1)*1.0/pixsize[0] #get 250 scale in terms of pixel scale of map


prior250.set_prf(prf250.array,pind250,pind250)

prior250.get_pointing_matrix()


fit=xid_mod.lstdrv_stan_highz(prior250,iter=1500)
posterior=xid_mod.posterior_stan(fit,prior250.nsrc)

outfile=output_folder+'COSMOS_psw_Fit_'+str(mu_sim)+'_'+str(sig_sim)+'.pkl'
#outfile=output_folder+'Lacey_rbandcut_19_8_log_flux.pkl'
with open(outfile, 'wb') as f:
    pickle.dump({'psw':prior250,'posterior':posterior},f)

