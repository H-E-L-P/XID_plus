
# coding: utf-8

# In[1]:

import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import proto_XIDp_mod as xid_mod


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


# In[6]:

sx250,sy250,sra,sdec,n_src,sgood=xid_mod.cat_check_convert(inra,indec,w_250) 


# In[7]:

#get positions of sources in terms of pixels for other two maps
sx350,sy350=w_350.wcs_world2pix(sra,sdec,0)#still not convinced about origin. is it 0 or 1
sx500,sy500=w_500.wcs_world2pix(sra,sdec,0)#still not convinced about origin. is it 0 or 1


# In[8]:

#-----250-------------
bad=np.logical_or(np.logical_or
                  (np.invert(np.isfinite(im250)),
                   np.invert(np.isfinite(nim250))),(nim250 == 0))
if(bad.sum() >0):
    im250[bad]=0.
    nim250[bad]=1.
#-----350-------------
bad=np.logical_or(np.logical_or
                  (np.invert(np.isfinite(im350)),
                   np.invert(np.isfinite(nim350))),(nim350 == 0))
if(bad.sum() >0):
    im350[bad]=0.
    nim350[bad]=1.
#-----500-------------
bad=np.logical_or(np.logical_or
                  (np.invert(np.isfinite(im500)),
                   np.invert(np.isfinite(nim500))),(nim500 == 0))
if(bad.sum() >0):
    im500[bad]=0.
    nim500[bad]=1.


# Load up High Z sources for stacking

# In[9]:

#Folder containing prior input catalogue
folder="'/research/astro/fir/HELP/high_z/"
#prior catalogue
prior_cat="ZSOUTHDEEP.fits"
hdulist = fits.open(folder+prior_cat)
fcat_z=hdulist[1].data
hdulist.close()
inra_z=fcat_z['RA']
indec_z=fcat_z['DEC']
nrealcat_z=fcat.size


# In[10]:

sx250_z,sy250_z,sra_z,sdec_z,n_src_z,sgood_z=xid_mod.cat_check_convert(inra_z,indec_z,w_250)


# Sort out PRF

# In[11]:

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


# In[12]:

sx250_tot=np.append(sx250,sx250_z)
sy250_tot=np.append(sy250,sy250_z)


# In[13]:

x_pix,y_pix=np.meshgrid(np.arange(0,w_250._naxis1),np.arange(0,w_250._naxis2))


# In[14]:

#get pointing matrix
amat_data,amat_row,amat_col,A,sx_pix,sy_pix,snoisy_map,ssig_map,snsrc,snpix=xid_mod.lstdrv_initsolveSP(sx250_tot
                                                                                                   ,sy250_tot,prf250,im250,nim250,x_pix,y_pix)


# In[ ]:

fit_data,chains,iter=xid_mod.lstdrv_stan_highz(amat_data,amat_row,amat_col,snoisy_map,ssig_map,n_src,n_src_z,snpix,bkg250,3.0)


# In[ ]:

output_folder='/research/astro/fir/HELP/XID_plus_output/'
outfile=output_folder+'goodss_highz_fit.pkl'
with open(outfile, 'wb') as f:
            pickle.dump({'A':A,'chains':fit_data,'x_pix':sx_pix,'y_pix':sy_pix,'sig_pix':snoisy_map,'im_pix':ssig_map,'snsrc':snsrc,'snpix':snpix}, f)

