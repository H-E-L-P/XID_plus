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
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'

with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)

tiles=obj['tiles']
tiling_list=obj['tiling_list']
nsources=tiling_list.shape[0]
sources_percentile=np.empty((nsources,14))
#hdulist_master=xid_mod.create_empty_XIDp_SPIREcat(nsources)
stan_fit_master=np.empty((500,4,(nsources+1.0)*3))
for i in np.arange(0,len(tiles)):
    print 'on tile '+str(i)+' of '+str(len(tiles))
    tile=tiles[i]
    #find which sources from master list are we interested in
    ind= (np.around(tiling_list[:,2],3) == np.around(tile[0,0],3)) & (np.around(tiling_list[:,3],3) == np.around(tile[1,0],3))
    if ind.sum() >0:
	    infile=output_folder+'Lacey_log10_norm0_5_'+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')+'.pkl'
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
	    
	    #hdulist=xid_mod.create_XIDp_SPIREcat(posterior,prior250,prior350,prior500)
	    #table=hdulist[1].data
	    #hdulist.writeto(output_folder+'lacy_rband_19_8_normal_log10fluxprior_'+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')+'.fits')

	    #match interested sources to those fitted
	    #c = SkyCoord(ra=table['ra']*u.degree, dec=table['dec']*u.degree)
	    c = SkyCoord(ra=prior250.sra*u.degree, dec=prior250.sdec*u.degree)

	    c2 = SkyCoord(ra=tiling_list[ind,0]*u.degree, dec=tiling_list[ind,1]*u.degree)
	    #get indices in prior list which match those in master list
	    idx, d2d, d3d = c2.match_to_catalog_sky(c)
	    #for i in range(len(hdulist_master[1].data.columns)):
		    
		#    hdulist_master[1].data.field(i)[ind]=table.field(i)[idx]

	    #sources_percentile[ind,:]=table[idx][1:]

	    #---fill in master stan_fit_array
	    stan_ind_oneband=np.append(ind,False)#don't add background
	    stan_ind=np.append(stan_ind_oneband,stan_ind_oneband)#adjoin psw and pmw
	    stan_ind=np.append(stan_ind,stan_ind_oneband)#adjoin with plw

	    idx_allbands=np.append(np.append(idx,prior250.nsrc+1+idx),2*prior250.nsrc+2+idx)
	    print stan_fit_master[:,:,stan_ind].shape,posterior.stan_fit[:,:,idx_allbands].shape,ind.sum(),idx.size
	    stan_fit_master[:,:,stan_ind]=posterior.stan_fit[:,:,idx_allbands]




#-------------since wcs info isnt saving properly, get it from original maps---------
#Folder containing maps
imfolder='/research/astro/fir/cclarke/lacey/released/'
#field
field='COSMOS'
#SMAP version
SMAPv='4.2'

pswfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PSW_hipe.fits.gz'#SPIRE 250 map
pmwfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PMW_hipe.fits.gz'#SPIRE 350 map
plwfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PLW_hipe.fits.gz'#SPIRE 500 map

#-----250-------------
hdulist = fits.open(pswfits)
w_250 = wcs.WCS(hdulist[1].header)
pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----350-------------
hdulist = fits.open(pmwfits)
w_350 = wcs.WCS(hdulist[1].header)
pixsize350=3600.0*w_350.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()
#-----500-------------
hdulist = fits.open(plwfits)
w_500 = wcs.WCS(hdulist[1].header)
pixsize500=3600.0*w_500.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()



#----------------------------------------------------------------------------------


prior_cat_file='lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test.fits'
#---------create master classes------------------------
prior250_master=xid_mod.prior(prior250.im,prior250.nim,w_250,prior250.imphdu)#Initialise with map, uncertianty map, wcs info and primary header
#print help(prior250_master.prior_cat)
prior250_master.prior_cat(tiling_list[:,0],tiling_list[:,1],prior_cat_file)
prior350_master=xid_mod.prior(prior350.im,prior350.nim,w_350,prior350.imphdu)#Initialise with map, uncertianty map, wcs info and primary header
prior350_master.prior_cat(tiling_list[:,0],tiling_list[:,1],prior_cat_file)
prior500_master=xid_mod.prior(prior500.im,prior500.nim,w_500,prior500.imphdu)#Initialise with map, uncertianty map, wcs info and primary header
prior500_master.prior_cat(tiling_list[:,0],tiling_list[:,1],prior_cat_file)

posterior_master=xid_mod.posterior_stan(stan_fit_master,nsources)

with open(output_folder+'Tiled_master_Lacey_notlog_flux_norm0_5.pkl', 'wb') as f:
    pickle.dump({'psw':prior250_master,'pmw':prior350_master,'plw':prior500_master,'posterior':posterior_master},f)



#hdulist_master.writeto(output_folder+'Tiled_SPIRE_cat_flux_notlog.fits')




    
