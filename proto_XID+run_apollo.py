
# coding: utf-8

## XID high level code

# import modules

# In[1]:

import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import proto_XIDp_mod as xid_mod



# In[6]:

#Folder containing maps
imfolder='/research/astrodata/fir/hermes/xid/XID_2014/SMAP_images_v4.2/'
#field
field='COSMOS'
#SMAP version
SMAPv='4.2'


garea=0.25


# In[7]:

pswfits=imfolder+field+'_image_250_SMAP_v'+SMAPv+'.fits'#SPIRE 250 map
pmwfits=imfolder+field+'_image_350_SMAP_v'+SMAPv+'.fits'#SPIRE 350 map
plwfits=imfolder+field+'_image_500_SMAP_v'+SMAPv+'.fits'#SPIRE 500 map


# In[8]:

#Folder containing prior input catalogue
folder="/research/astrodata/fir/hermes/www/xid/rel0712"
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


# Open images and noise maps and use WCS module in astropy to get header information

# In[9]:

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



# Filter out sources not in the map (just check 250)

# In[10]:

#get positions of sources in terms of pixels
sx250,sy250=w_250.wcs_world2pix(inra,indec,0)#still not convinced about origin. is it 0 or 1


# In[11]:

#check if sources are within map and if the nearest pixel has a finite value 
sgood=(sx250 > 0) & (sx250 < w_250._naxis1) & (sy250 > 0) & (sy250 < w_250._naxis2) & np.isfinite(im250[np.rint(sx250).astype(int),np.rint(sy250).astype(int)])#this gives boolean array for cat
#Redefine prior list so it only contains sources in the map
sx250=sx250[sgood]
sy250=sy250[sgood]
sra=inra[sgood]
sdec=indec[sgood]
p_src=f_src[sgood]
n_src=sgood.sum()


# Since I am only testing, redo this so that I only fit sources within a given range of the mean pixel position of the prior list

# In[12]:

#define range
sx_mean=np.mean(sx250)
sy_mean=np.mean(sy250)
p_range=200
#check if sources are within range and if the nearest pixel has a finite value 

sgood=(sx250 > sx_mean-p_range) & (sx250 < sx_mean+p_range) & (sy250 > sy_mean-p_range) & (sy250 < sy_mean+p_range) & np.isfinite(im250[np.rint(sx250).astype(int),np.rint(sy250).astype(int)])#this gives boolean array for cat
#Redefine prior list so it only contains sources in the map
sx250=sx250[sgood]
sy250=sy250[sgood]
sra=sra[sgood]
sdec=sdec[sgood]
p_src=p_src[sgood]
n_src=sgood.sum()
print n_src,sx250.shape


# In[13]:

#get positions of sources in terms of pixels for other two maps
sx350,sy350=w_350.wcs_world2pix(sra,sdec,0)#still not convinced about origin. is it 0 or 1
sx500,sy500=w_500.wcs_world2pix(sra,sdec,0)#still not convinced about origin. is it 0 or 1


# For pixels in the map that are NaNs or where error is zero, set flux to zero and error to 1

# In[14]:

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


# Point response information, at the moment its 2D Gaussian, but should be general. All lstdrv_solvfluxes needs is 2D array with prf

# In[15]:

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
print pixsize
print pfwhm[0]/2.355
prf250=Gaussian2DKernel(pfwhm[0]/2.355,x_size=paxis[0],y_size=paxis[1])
prf250.normalize(mode='peak')
prf350=Gaussian2DKernel(pfwhm[1]/2.355,x_size=paxis[0],y_size=paxis[1])
prf350.normalize(mode='peak')
prf500=Gaussian2DKernel(pfwhm[2]/2.355,x_size=paxis[0],y_size=paxis[1])
prf500.normalize(mode='peak')


# I need to work out what needs to be stored in the XID+ ouput catalogue

# In[16]:


xid_id=np.arange(0,nrealcat,dtype=long)[sgood]
print n_src
tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='xid', format='I', array=xid_id),fits.Column(name='inra', format='E',array=sra),fits.Column(name='indec', format='E', array=sdec),fits.Column(name='oldf250', format='E', array=p_src),fits.Column(name='flux250', format='E', array=np.zeros(n_src)),fits.Column(name='el250', format='E', array=np.zeros(n_src)),fits.Column(name='eu250', format='E', array=np.zeros(n_src)),fits.Column(name='R', format='E', array=np.zeros(n_src))])
prihdr = fits.Header()
prihdr['FIELD'] = field
prihdr['SMAPv'] = SMAPv
prihdr['Prior_C'] = prior_cat
prihdu = fits.PrimaryHDU(header=prihdr)


thdulist = fits.HDUList([prihdu, tbhdu])
print thdulist[1].data['flux250'].size


# In[17]:


rmap250,rmap_250old,fit_data,thdulist=xid_mod.lstdrv_solvefluxes(sx250,sy250,sx350,sy350,sx500,sy500,
                   prf250,prf350,prf500,
                   im250,im350,im500,nim250,nim350,nim500,
                   w_250,w_350,w_500,p_src,thdulist,bkg250,bkg350,bkg500,0.2*np.abs(bkg250),0.2*np.abs(bkg350),0.2*np.abs(bkg500))#wcs information
#save XID cat
output_folder='/research/astro/fir/HELP/XID_plus_output/'
thdulist.writeto(output_folder+'XID+_'+field+'_dat.fits')




# In[18]:

hdulist = fits.open(pswfits)
#copy over format for residual map
hdu_250list=hdulist[0:3]
hdu_250list[1].data=rmap250.T/1.0E3
hdu_250list[2].data=(im250-rmap250.T)/1.0E3
hdu_250list.writeto(output_folder+'XID+fit_map_'+field+'_dat.fits')
hdulist.close()
hdu_250list.close()




