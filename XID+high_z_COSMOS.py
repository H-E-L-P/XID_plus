
# coding: utf-8

# In[1]:

import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import XIDp_mod_beta as xid_mod
import pickle
import dill




#Folder containing maps
imfolder='/research/astrodata/fir/hermes/xid/XID_2014/SMAP_images_v4.2/' 
#field
field='COSMOS'
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

#Folder containing prior input catalogue
folder="/research/astrodata/fir/hermes/www/xid/rel0712/"
#prior catalogue
prior_cat="mod_cosmos-xid-pepprior-0512.fits.gz"
hdulist = fits.open(folder+prior_cat)
fcat=hdulist[1].data
hdulist.close()
inra=fcat['INRA']
indec=fcat['INDEC']
f_src=fcat['F250']
df_src=f_src
nrealcat=fcat.size
bkg250=fcat['bkg250'][0]
bkg350=fcat['bkg350'][0]
bkg500=fcat['bkg500'][0]

# Since I am only testing, redo this so that I only fit sources within a given range of the mean ra and dec position of the prior list

# In[12]:

#define range
ra_mean=np.mean(inra)
dec_mean=np.mean(indec)
p_range=0.125
#check if sources are within range and if the nearest pixel has a finite value 

sgood=(inra > ra_mean-p_range) & (inra < ra_mean+p_range) & (indec > dec_mean-p_range) & (indec < dec_mean+p_range)
inra=inra[sgood]
indec=indec[sgood]
n_src=sgood.sum()
print 'fitting '+str(n_src)+' sources'


# Load up High Z sources for stacking

# In[9]:

#Folder containing prior input catalogue
folder="/research/astro/fir/HELP/high_z/"
#prior catalogue
prior_cat_stack="g_mcut.fits"
hdulist = fits.open(folder+prior_cat_stack)
fcat_z=hdulist[1].data
hdulist.close()
inra_z=fcat_z['RA']
indec_z=fcat_z['DEC']
nrealcat_z=fcat.size


sgood_z=(inra_z > ra_mean-p_range) & (inra_z < ra_mean+p_range) & (indec_z > dec_mean-p_range) & (indec_z < dec_mean+p_range)
inra_z=inra_z[sgood_z]
indec_z=indec_z[sgood_z]
n_src=sgood_z.sum()
print 'fitting '+str(n_src)+' high z sources'

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

##---------fit using Gaussian beam-----------------------
prf250=Gaussian2DKernel(prfsize[0]/2.355,x_size=101,y_size=101)
prf250.normalize(mode='peak')
prf350=Gaussian2DKernel(prfsize[1]/2.355,x_size=101,y_size=101)
prf350.normalize(mode='peak')
prf500=Gaussian2DKernel(prfsize[2]/2.355,x_size=101,y_size=101)
prf500.normalize(mode='peak')
pind250=np.arange(0,101,1)*1.0/pixsize[0] #get 250 scale in terms of pixel scale of map
pind350=np.arange(0,101,1)*1.0/pixsize[1] #get 350 scale in terms of pixel scale of map
pind500=np.arange(0,101,1)*1.0/pixsize[2] #get 500 scale in terms of pixel scale of map
prior250=xid_mod.prior(im250,nim250,w_250,im250phdu)
prior250.prior_bkg(bkg250,2.0)
prior250.prior_cat(inra,indec,prior_cat)
prior250.prior_cat_stack(inra_z,indec_z,prior_cat_stack)
prior250.set_prf(prf250.array,pind250,pind250)
prior250.get_pointing_matrix()



fit_data,chains,iter=xid_mod.lstdrv_stan_highz(prior250)


# In[ ]:

output_folder='/research/astro/fir/HELP/XID_plus_output/'
outfile=output_folder+'COSMOS_g_mcut_fit_250_neg.pkl'
with open(outfile, 'wb') as f:
            pickle.dump({'prior250':prior250,'fit':fit_data}, f)

