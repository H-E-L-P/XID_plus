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
output_folder='/research/astro/fir/HELP/XID_plus_output/Tiling/'

with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)

tiles=obj['tiles']
tiling_list=obj['tiling_list']
nsources=tiling_list.shape[0]
sources_percentile=np.empty((nsources,14))
for i in np.arange(0,len(tiles)):
    tile=tiles[i]
    infile=output_folder+'lacy_rband_19_8_normal_log10fluxprior_'+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')+'.pkl'
    with open(infile, "rb") as f:
        dictname = pickle.load(f)
    prior250=dictname['psw']
    posterior=dictname['obj']
    hdulist=xid_mod.create_XIDp_cat(posterior,prior250,prior350,prior500)
    table=hdulist[1]
    #find which sources from master list are we interested in
    ind= (tiling_list[:,2] == tile[0,0]) & (tiling_list[:,3] == tile[0,1])
    
    #match interested sources to those fitted
    c = SkyCoord(ra=table['ra']*u.degree, dec=table['dec']*u.degree)
    c2 = SkyCoord(ra=tiling_list[ind,0]*u.degree, dec=tiling_list[ind,1]*u.degree)
    #get indices in prior list which match those in master list
    idx, d2d, d3d = c2.match_to_catalog_sky(c)
    sources_percentiles[ind,:]=table[idx][1:]
with open(output_folder+'combined_tiles_array.pkl', 'wb') as f:
    pickle.dump({'cat':sources_percentiles},f)



    
