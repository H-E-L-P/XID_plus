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
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/'

with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)
tiles=obj['tiles']
tiling_list=obj['tiling_list']
nsources=tiling_list.shape[0]
sources_percentile=np.empty((nsources,14))

for i in np.arange(0,len(tiles)):
	print 'on tile '+str(i)+' of '+str(len(tiles))
	tile=tiles[i]

	infile=output_folder+'lacy_log_uniform_prior_'+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')+'.pkl'

	with open(infile, "rb") as f:
	    dictname = pickle.load(f)
	prior250=dictname['psw']
	prior350=dictname['pmw']    
	prior500=dictname['plw']
	print '----wcs----'
	print prior250.wcs._naxis1,prior250.wcs._naxis2
	print '---------------'
	posterior=dictname['posterior']
	posterior.stan_fit=np.power(10.0,posterior.stan_fit)
	posterior.stan_fit[:,:,[prior250.nsrc,2*prior250.nsrc+1,3*prior250.nsrc+1]]=np.log10(posterior.stan_fit[:,:,[prior250.nsrc,2*prior250.nsrc+1,3*prior250.nsrc+2]])



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






	def yrep_map(prior,fvec):
	    from scipy.sparse import coo_matrix


	    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
	    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
	    f=coo_matrix((fvec, (range(0,prior.nsrc+1),np.zeros(prior.nsrc+1))), shape=(prior.nsrc+1, 1))
	    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc+1))
	    rmap_temp=(A*f)
	    pred_map=np.empty_like(prior.im)
	    pred_map[:,:]=prior.im
	    pred_map[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)#+np.random.randn(prior.snpix)*prior.snim

	    return pred_map
	samples,chains,params=posterior.stan_fit.shape
	flattened_post=posterior.stan_fit.reshape(samples*chains,params)

	#import matplotlib
	#matplotlib.use('PS')
	#import pylab as plt
	output_folder_maps='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior/'
	for i in range(0,50):#samples*chains):
	    print 'making map '+ str(i) 
	    pred_map=yrep_map(prior250,flattened_post[i,0:prior250.nsrc+1])
	    try:
		    ra_pix,dec_pix=w_250.wcs_pix2world(prior250.sx_pix,prior250.sy_pix,0)
		    sgood=(ra_pix > prior250.tile[0,0]) & (ra_pix < prior250.tile[0,2]+prior250.buffer_size) & (dec_pix > prior250.tile[1,0]-prior250.buffer_size) & (dec_pix < prior250.tile[1,2]+prior250.buffer_size)#
	    
	    #plt.imshow(pred_map/1.0E03,interpolation='nearest',vmin=-1E-3,vmax=0.5E-1)
		    res = fits.open(output_folder_maps+'maps/res250_'+str(i)+'.fits',mode='update')
	    #plt.savefig(output_folder+'maps/SMAP250_'+str(i)+'.eps')
		    res[1].data[prior250.sy_pix[sgood],prior250.sx_pix[sgood]]=(im250[prior250.sy_pix[sgood],prior250.sx_pix[sgood]]/1.0E3)-(pred_map[prior250.sy_pix[sgood],prior250.sx_pix[sgood]]/1.0E03)
		    res.flush()
		    res.close()
	    except ValueError:
		    print ' oh oh'
