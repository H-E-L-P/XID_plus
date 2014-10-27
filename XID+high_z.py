
# coding: utf-8

# In[1]:

import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import XIDp_mod_beta as xid_mod


# In[2]:

#Folder containing maps
imfolder='/research/astrodata/fir/hermes/xid/XID_2014/SMAP_images_v4.2/' 
#field
field='GOODS-S'
#SMAP version
SMAPv='4.2'


# In[3]:

pswfits=imfolder+field+'_image_250_SMAP_v'+SMAPv+'.fits'#SPIRE 250 map
pmwfits=imfolder+field+'_image_350_SMAP_v'+SMAPv+'.fits'#SPIRE 350 map
plwfits=imfolder+field+'_image_500_SMAP_v'+SMAPv+'.fits'#SPIRE 500 map


# In[4]:

#-----250-------------
hdulist = fits.open(pswfits)
im250=hdulist[1].data*1.0E3
nim250=hdulist[2].data*1.0E3
w_250 = wcs.WCS(hdulist[1].header)
im250phdu=hdulist[0]
pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----350-------------
hdulist = fits.open(pmwfits)
im350=hdulist[1].data*1.0E3
nim350=hdulist[2].data*1.0E3
w_350 = wcs.WCS(hdulist[1].header)
pixsize350=3600.0*w_350.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----500-------------
hdulist = fits.open(plwfits)
im500=hdulist[1].data*1.0E3
nim500=hdulist[2].data*1.0E3
w_500 = wcs.WCS(hdulist[1].header)
pixsize500=3600.0*w_500.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()


# In[5]:

#Folder containing prior input catalogue
folder="/research/astrodata/fir/hermes/xid/XID_2014/goodss/"
#prior catalogue
prior_cat="cat_prior_24_goodss.fits"
hdulist = fits.open(folder+prior_cat)
fcat=hdulist[1].data
hdulist.close()
inra=fcat['RA_IRAC']
indec=fcat['DEC_IRAC']
nrealcat=fcat.size
bkg250=-2.0
bkg350=-2.0
bkg500=-2.0






# Load up High Z sources for stacking

# In[9]:

#Folder containing prior input catalogue
folder="/research/astro/fir/HELP/high_z/"
#prior catalogue
prior_cat_stack="ZSOUTHDEEP.fits"
hdulist = fits.open(folder+prior_cat_stack)
fcat_z=hdulist[1].data
hdulist.close()
inra_z=fcat_z['RA']
indec_z=fcat_z['DEC']
nrealcat_z=fcat.size


# In[10]:




#pixsize array (size of pixels in arcseconds)
pixsize=np.array([pixsize250,pixsize350,pixsize500])
#point response function for the three bands
prfsize=np.array([18.15,25.15,36.3])
#set fwhm of prfs in terms of pixels
pfwhm=prfsize/pixsize
#set size of prf array (in pixels)
paxis=[13,13]
#use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
from astropy.convolution import Gaussian2DKernel

prf250=Gaussian2DKernel(pfwhm[0]/2.355,x_size=paxis[0],y_size=paxis[1])
prf250.normalize(mode='peak')
prf350=Gaussian2DKernel(pfwhm[1]/2.355,x_size=paxis[0],y_size=paxis[1])
prf350.normalize(mode='peak')
prf500=Gaussian2DKernel(pfwhm[2]/2.355,x_size=paxis[0],y_size=paxis[1])
prf500.normalize(mode='peak')

prior250=xid_mod.prior(prf250,im250,nim250,w_250,im250phdu)
prior250.prior_bkg(bkg250,2.0)
prior250.prior_cat(inra,indec,prior_cat)
prior250.prior_cat_stack(inra_z,indec_z,prior_cat_stack)
prior250.get_pointing_matrix()
print prior250.amat_data.size
print prior250.amat_row.astype(long).size


fit_data,chains,iter=xid_mod.lstdrv_stan_highz(prior250)


# In[ ]:

output_folder='/research/astro/fir/HELP/XID_plus_output/'
outfile=output_folder+'goodss_highz_fit_250.pkl'
with open(outfile, 'wb') as f:
            pickle.dump({'prior':prior250,'fit':fit_data}, f)

