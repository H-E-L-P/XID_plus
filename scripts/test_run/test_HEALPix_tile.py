__author__ = 'pdh21'
import numpy as np
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import sys
sys.path.append('/Users/pdh21/HELP/XID_plus/')
import xidplus


#Folder containing maps
imfolder='../../test_files/'
#field
field='COSMOS'
#SMAP version
SMAPv='4.2'


# In[7]:

pswfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PSW_hipe.fits.gz'#SPIRE 250 map
pmwfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PMW_hipe.fits.gz'#SPIRE 350 map
plwfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PLW_hipe.fits.gz'#SPIRE 500 map


#----output folder-----------------
output_folder='./'

#Folder containing prior input catalogue
folder='../../test_files/'
#prior catalogue
prior_cat='lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test.fits'

hdulist = fits.open(folder+prior_cat)
fcat=hdulist[1].data
hdulist.close()
inra=fcat['RA']
indec=fcat['DEC']
f_src=fcat['S100']#100 micron flux
df_src=f_src
nrealcat=fcat.size
bkg250=0
bkg350=0
bkg500=0

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
hdulist = fits.open(pmwfits)
im350phdu=hdulist[0].header
im350hdu=hdulist[1].header

im350=hdulist[1].data*1.0E3
nim350=hdulist[2].data*1.0E3
w_350 = wcs.WCS(hdulist[1].header)
pixsize350=3600.0*w_350.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----500-------------
hdulist = fits.open(plwfits)
im500phdu=hdulist[0].header
im500hdu=hdulist[1].header
im500=hdulist[1].data*1.0E3
nim500=hdulist[2].data*1.0E3
w_500 = wcs.WCS(hdulist[1].header)
pixsize500=3600.0*w_500.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()

#pixsize array (size of pixels in arcseconds)
pixsize=np.array([pixsize250,pixsize350,pixsize500])
#point response function for the three bands
prfsize=np.array([18.15,25.15,36.3])
#use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
from astropy.convolution import Gaussian2DKernel


#Set prior classes
#---prior250--------
prior250=xidplus.prior(im250,nim250,im250phdu,im250hdu)#Initialise with map, uncertianty map, wcs info and primary header
prior250.prior_cat(inra,indec,prior_cat)#Set input catalogue
prior250.prior_bkg(-10.0,3)
#prior250.moc.write('/Users/pdh21/HELP/XID_plus/scripts/test_run/MOC_master.fits')


print prior250.snpix,prior250.nsrc,prior250.sra.size,prior250.sx.size

#from moc, get healpix pixels at a given order
from xidplus import moc_routines

tiles=moc_routines.get_HEALPix_pixels(9,prior250.sra,prior250.sdec,unique=True)
print len(tiles)
moc=moc_routines.get_fitting_region(9,tiles[50])
#moc.write('/Users/pdh21/HELP/XID_plus/scripts/test_run/MOC_tile_20.fits')

prior250.set_tile(moc)
prior250.moc.write('/Users/pdh21/HELP/XID_plus/scripts/test_run/MOC_tile_20_fit.fits')

#pixsize array (size of pixels in arcseconds)
pixsize=np.array([pixsize250,pixsize350,pixsize500])
#point response function for the three bands
prfsize=np.array([18.15,25.15,36.3])
#use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
from astropy.convolution import Gaussian2DKernel
prf250=Gaussian2DKernel(prfsize[0]/2.355,x_size=101,y_size=101)
prf250.normalize(mode='peak')
pind250=np.arange(0,101,1)*1.0/pixsize[0] #get 250 scale in terms of pixel scale of map
prior250.set_prf(prf250.array,pind250,pind250)

print prior250.nsrc
prior250.get_pointing_matrix()
prior250.upper_lim_map()
print prior250.prior_flux_upper