import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import XIDp_mod_beta as xid_mod
import os
import sys

#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/uniform_prior/'

with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)
tiles=obj['tiles']
tiling_list=obj['tiling_list']
nsources=tiling_list.shape[0]
sources_percentile=np.empty((nsources,14))


with open(output_folder+'Tiled_master_Lacey_notlog_flux.pkl',, "rb") as f:
        obj = pickle.load(f)
prior250=obj['psw']
prior350=obj['pmw']    
prior500=obj['plw']

posterior=obj['posterior']

#flatten chains------
samples,chains,params=posterior.stan_fit.shape
flattened_post=posterior.stan_fit.reshape(samples*chains,params)



#Folder containing maps
imfolder='/research/astro/fir/cclarke/lacey/released/'
#field
field='COSMOS'
#SMAP version
SMAPv='4.2'


# In[7]:

pswfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PSW_hipe.fits.gz'#SPIRE 250 map
#-----250-------------
hdulist = fits.open(pswfits)
im250phdu=hdulist[0].header
fits_template=hdulist[1]
im250=hdulist[1].data*1.0E3
nim250=hdulist[2].data*1.0E3
w_250 = wcs.WCS(hdulist[1].header)
pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()


#calculate pointing matrix
##---------fit using Gaussian beam-----------------------
pixsize=np.array([pixsize250,pixsize350,pixsize500])
#point response function for the three bands
prfsize=np.array([18.15,25.15,36.3])


from astropy.convolution import Gaussian2DKernel
prf250=Gaussian2DKernel(prfsize[0]/2.355,x_size=101,y_size=101)
prf250.normalize(mode='peak')
pind250=np.arange(0,101,1)*1.0/pixsize[0] #get 250 scale in terms of pixel scale of map
prior250.set_prf(prf250.array,pind250,pind250)
prior250.get_pointing_matrix()




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

for i in range(0,samples*chains):
    pred_map=yrep_map(prior250,flattened_posterior[i,:])
    fits_template.data=predmap
    fits_template.writeto(output_folder+'maps/SMAP250_'+str(i)+'.fits')
    
