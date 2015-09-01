import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import wcs
import pickle
import dill
import sys
import XIDp_mod_beta as xid_mod


#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/'

with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)

tiles=obj['tiles']
tiling_list=obj['tiling_list']
nsources=tiling_list.shape[0]

for i in np.arange(0,len(tiles)):
    print 'on tile '+str(i)+' of '+str(len(tiles))
    tile=tiles[i]
    #find which sources from master list are we interested in
    ind= (np.around(tiling_list[:,2],3) == np.around(tile[0,0],3)) & (np.around(tiling_list[:,3],3) == np.around(tile[1,0],3))
    if ind.sum() >0:
	    
            filename='lacy_log_uniform_prior_'+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')
	    with open(output_folder+filename+'.pkl', "rb") as f:
		dictname = pickle.load(f)
	    prior250=dictname['psw']
	    prior350=dictname['pmw']    
	    prior500=dictname['plw']
	    posterior=dictname['posterior']
            thdulist=xid_mod.create_XIDp_SPIREcat_nocov(posterior,prior250,prior350,prior500)
            thdulist.writeto(output_folder+'catalogues/'+filename+'.fits')
             #match interested sources to those fitted
	    #c = SkyCoord(ra=table['ra']*u.degree, dec=table['dec']*u.degree)
	    c = SkyCoord(ra=prior250.sra*u.degree, dec=prior250.sdec*u.degree)

	    c2 = SkyCoord(ra=tiling_list[ind,0]*u.degree, dec=tiling_list[ind,1]*u.degree)
	    #get indices in prior list which match those in master list
	    idx, d2d, d3d = c2.match_to_catalog_sky(c)
            try:
                master_table=vstack([master_table,Table(thdulist[1].data[idx])])
            except NameError:
                master_table=Table(thdulist[1].data[idx])
            thdulist.close()
#master_table.write(output_folder+'catalogues/'+'master_catalogue.fits')
