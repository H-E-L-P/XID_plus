import numpy as np
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import sys
import os
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
bkg250=-5
bkg350=-5
bkg500=-5

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


#only use sources that have a flux of greater than 50 microjanskys at 100 microns


sgood=f_src >0.050

inra=inra[sgood]
indec=indec[sgood]
n_src=sgood.sum()



# Point response information, at the moment its 2D Gaussian,

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
prior250.prior_bkg(bkg250,5)#Set prior on background
#---prior350--------
prior350=xidplus.prior(im350,nim350,im350phdu,im350hdu)
prior350.prior_cat(inra,indec,prior_cat)
prior350.prior_bkg(bkg350,5)

#---prior500--------
prior500=xidplus.prior(im500,nim500,im500phdu,im500hdu)
prior500.prior_cat(inra,indec,prior_cat)
prior500.prior_bkg(bkg500,5)

print 'fitting '+ str(prior250.nsrc)+' sources \n'


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

prior250.set_prf(prf250.array,pind250,pind250)
prior350.set_prf(prf350.array,pind350,pind350)
prior500.set_prf(prf500.array,pind500,pind500)

#from moc, get healpix pixels at a given order
from xidplus import moc_routines
order=12
tiles=moc_routines.get_HEALPix_pixels(order,prior250.sra,prior250.sdec,unique=True)

try:
    if sys.argv[1] == 'Master':
        print '----- There are '+str(len(tiles))+' tiles required for input catalogue'
        outfile=output_folder+'Master_prior.pkl'
        with open(outfile, 'wb') as f:
            pickle.dump({'psw':prior250,'pmw':prior350,'plw':prior500,'tiles':tiles,'order':order},f)
        raise SystemExit()


except:
    pass

try:
    taskid = np.int(os.environ['SGE_TASK_ID'])
    task_first=np.int(os.environ['SGE_TASK_FIRST'])
    task_last=np.int(os.environ['SGE_TASK_LAST'])

except KeyError:
    print "Error: could not read SGE_TASK_ID from environment"
    taskid = int(raw_input("Please enter task id: "))
    print "you entered", taskid


moc=moc_routines.get_fitting_region(order,tiles[taskid-1])
prior250.set_tile(moc)
prior350.set_tile(moc)
prior500.set_tile(moc)

print 'fitting '+ str(prior250.nsrc)+' sources \n'
print 'there are '+ str(prior250.snpix)+' pixels'

prior250.get_pointing_matrix()
prior350.get_pointing_matrix()
prior500.get_pointing_matrix()

print 'set prior upper limit'
prior250.upper_lim_map()
prior250.lower_lim_flux(-2)
prior350.upper_lim_map()
prior350.lower_lim_flux(-2)
prior500.upper_lim_map()
prior500.lower_lim_flux(-2)


from xidplus.stan_fit import SPIRE
fit=SPIRE.all_bands(prior250,prior350,prior500,iter=1500)
posterior=xidplus.posterior_stan(fit,[prior250,prior350,prior500])

outfile=output_folder+'Lacy_test_file_'+str(tiles[taskid-1])+'_'+str(order)+'.pkl'
with open(outfile, 'wb') as f:
   pickle.dump({'psw':prior250,'pmw':prior350,'plw':prior500,'posterior':posterior},f)

