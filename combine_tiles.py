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
hdulist_master=xid_mod.create_empty_XIDp_SPIREcat(nsources)
for i in np.arange(0,len(tiles)):
    print 'on tile '+str(i)+' of '+str(len(tiles))
    tile=tiles[i]
    #find which sources from master list are we interested in
    ind= (np.around(tiling_list[:,2],3) == np.around(tile[0,0],3)) & (np.around(tiling_list[:,3],3) == np.around(tile[1,0],3))
    if ind.sum() >0:
	    infile=output_folder+'lacy_uniform_fluxprior_'+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')+'.pkl'
	    with open(infile, "rb") as f:
		dictname = pickle.load(f)
	    prior250=dictname['psw']
	    prior350=dictname['pmw']    
	    prior500=dictname['plw']

	    posterior=dictname['posterior']
	    posterior.stan_fit=np.power(10.0,posterior.stan_fit)
	    hdulist=xid_mod.create_XIDp_SPIREcat(posterior,prior250,prior350,prior500)
	    table=hdulist[1].data
	    #hdulist.writeto(output_folder+'lacy_rband_19_8_normal_log10fluxprior_'+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')+'.fits')

	    #match interested sources to those fitted
	    c = SkyCoord(ra=table['ra']*u.degree, dec=table['dec']*u.degree)
	    c2 = SkyCoord(ra=tiling_list[ind,0]*u.degree, dec=tiling_list[ind,1]*u.degree)
	    #get indices in prior list which match those in master list
	    idx, d2d, d3d = c2.match_to_catalog_sky(c)
	    for i in range(len(hdulist_master[1].data.columns)):
		    
		    hdulist_master[1].data.field(i)[ind]=table.field(i)[idx]
	    #sources_percentile[ind,:]=table[idx][1:]
hdulist_master.writeto(output_folder+'Tiled_SPIRE_cat_flux_notlog.fits')
#with open(output_folder+'combined_tiles_array.pkl', 'wb') as f:
    #pickle.dump({'cat':sources_percentile},f)



    
