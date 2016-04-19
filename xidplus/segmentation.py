import numpy as np
from sklearn.neighbors import NearestNeighbors
from astropy import units as u
from astropy.coordinates import SkyCoord


def Segmentation_scheme(inra,indec,inID,tile_l):
    """For a given prior catalogue, create a tiling scheme with given tile size. \n Returns tiles for which there are sources""" 
    ra_min=np.floor(10.0*np.min(inra))/10.0
    ra_max=np.floor(10.0*np.max(inra))/10.0
    dec_min=np.floor(10.0*np.min(indec))/10.0
    dec_max=np.floor(10.0*np.max(indec))/10.0

    #Create array to store optimum tile for each source
    tiling_list=np.recarray((inra.size,),dtype=[('ra',np.float64),('dec',np.float64),('ra_t',np.float64),('dec_t',np.float64),('dist',np.float64),('ID',np.str_,20)])
    #Create tiles
    tiles=[]
    tiling_list['dist']=tile_l
    for ra in np.arange(ra_min,ra_max,0.75*tile_l):
        for dec in np.arange(dec_min,dec_max,0.75*tile_l):
            #create tile for this ra and dec
            tile=np.array([[ra,dec],[ra+tile_l,dec],[ra+tile_l,dec+tile_l],[ra,dec+tile_l]]).T
            #check how many sources are in this tile
            sgood=(inra > tile[0,0]) & (inra < tile[0,1]) & (indec > tile[1,0]) & (indec < tile[1,2])

            if sgood.sum() >0:
                tiles.append(tile)
                #work out distance from tile centre to each source in tile
                dist=np.power(np.power((ra+tile_l/2.0)-inra[sgood],2)+np.power((dec+tile_l/2.0)-indec[sgood],2),0.5)
                ii=0
                for i in np.arange(0,inra.size)[sgood]:
                    #store ra and dec of optimum tile as well as distance
                    if tiling_list['dist'][i] > dist[ii]:
                        tiling_list[i]=(inra[i],indec[i],ra,dec,dist[ii],inID[i])
                    ii+=1
    return tiles, tiling_list

def make_master_catalogue(output_folder,tile_file_name):
    import pickle
    import dill
    import catalogue
    with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)
    
    
    tiles=obj['tiles']
    tiling_list=obj['tiling_list']
    nsources=tiling_list.size
    for i in np.arange(0,4):#len(tiles)):
        print 'on tile '+str(i)+' of '+str(len(tiles))
        tile=tiles[i]
        #find which sources from master list are we interested in
        ind= (np.around(tiling_list['ra_t'][:],3) == np.around(tile[0,0],3)) & (np.around(tiling_list['dec_t'][:],3) == np.around(tile[1,0],3))
        if ind.sum() >0:

                filename=tile_file_name+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')
                with open(output_folder+filename+'.pkl', "rb") as f:
                    dictname = pickle.load(f)
                prior250=dictname['psw']
                prior350=dictname['pmw']    
                prior500=dictname['plw']
                posterior=dictname['posterior']
                #match interested sources to those fitted
                idx=[]
                for j in range(0,ind.sum()):
                    idx.append(np.where(prior250.ID == tiling_list['ID'][ind][j])[0])
                thdulist=catalogue.create_XIDp_SPIREcat_post(posterior,prior250,prior350,prior500)
                thdulist[1].data=thdulist[1].data[idx]
                thdulist.writeto(output_folder+'catalogues/'+filename+'.fits')
                 #match interested sources to those fitted
                #try:
                #    master_table=vstack([master_table,Table(thdulist[1].data[idx])])
                #except NameError:
                #    master_table=Table(thdulist[1].data[idx])
                thdulist.close()
#master_table.write(output_folder+'catalogues/'+'master_catalogue.fits')

def make_master_posterior(output_folder,tile_file_name): 
    import pickle
    import dill
    with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)
    
    
    tiles=obj['tiles']
    tiling_list=obj['tiling_list']
    nsources=tiling_list.size
    stan_fit_master=np.empty((750,4,(nsources+1.0)*3))

    for i in np.arange(0,len(tiles)):
        print 'on tile '+str(i)+' of '+str(len(tiles))
        tile=tiles[i]
        #find which sources from master list are we interested in
        ind= (np.around(tiling_list['ra_t'][:],3) == np.around(tile[0,0],3)) & (np.around(tiling_list['dec_t'][:],3) == np.around(tile[1,0],3))
        if ind.sum() >0:

                filename=tile_file_name+str(tile[0,0]).replace('.','_')+'p'+str(tile[1,0]).replace('.','_')
                with open(output_folder+filename+'.pkl', "rb") as f:
                    dictname = pickle.load(f)
                prior250=dictname['psw']
                prior350=dictname['pmw']    
                prior500=dictname['plw']
                posterior=dictname['posterior']
                #match interested sources to those fitted
                idx=[]
                for j in range(0,ind.sum()):
                    idx.append(np.where(prior250.ID == tiling_list['ID'][ind][j])[0])
                idx=np.array(idx)
                 #---fill in master stan_fit_array
                stan_ind_oneband=np.append(ind,False)#don't add background
                stan_ind=np.append(stan_ind_oneband,stan_ind_oneband)#adjoin psw and pmw
                stan_ind=np.append(stan_ind,stan_ind_oneband)#adjoin with plw
                idx_allbands=np.append(np.append(idx,prior250.nsrc+1+idx),2*prior250.nsrc+2+idx)
                print stan_fit_master[:,:,stan_ind].shape,posterior.stan_fit[:,:,idx_allbands].shape,ind.sum(),len(idx)
                stan_fit_master[:,:,stan_ind]=posterior.stan_fit[:,:,idx_allbands]
    with open(output_folder+'master_posterior.pkl', 'wb') as f:
        pickle.dump({'posterior':stan_fit_master},f)

def make_master_posterior_HEALpix(output_folder,Master_filename,chains=4,iters=750):
    """function to combine a tiled run of XID+, based on the HEALPix pixelisation scheme"""
    #load in master posterior,
    #load up each tile, return sources in healpix pixel only: need a routine for this as will need to redo this numerous times
    #load up posterior array:
    import pickle
    import dill
    from xidplus import moc_routines
    with open(output_folder+Master_filename, "rb") as f:
        Master = pickle.load(f)

    tiles=Master['tiles']
    order=Master['order']
    prior250=Master['psw']
    stan_fit_master=np.empty((iters,chains,(prior250.nsrc+2.0)*3))
    for i in range(0,len(tiles)):
        print 'On tile '+str(i)+' out of '+str(len(tiles))
        infile=output_folder+'Lacy_test_file_'+str(tiles[i])+'_'+str(order)+'.pkl'
        with open(infile, "rb") as f:
            obj = pickle.load(f)
        tmp_prior250=obj['psw']
        tmp_prior350=obj['pmw']
        tmp_prior500=obj['plw']
        tmp_posterior=obj['posterior']

        #work out what sources in tile to keep
        kept_sources=moc_routines.sources_in_tile(tiles[i],order,tmp_prior250.sra,tmp_prior250.sdec)
        #create indices for posterior (i.e. inlcude backgrounds and sigma_conf)
        ind_tmp=np.array(kept_sources+[False]+kept_sources+[False]+kept_sources+[False]+[False,False,False])
        kept_sources=np.array(kept_sources)
        #scale from 0-1 to flux values:
        lower=np.append(np.append(tmp_prior250.prior_flux_lower[kept_sources],tmp_prior350.prior_flux_lower[kept_sources]),tmp_prior500.prior_flux_lower[kept_sources])
        upper=np.append(np.append(tmp_prior250.prior_flux_upper[kept_sources],tmp_prior350.prior_flux_upper[kept_sources]),tmp_prior500.prior_flux_upper[kept_sources])
        #work out what sources in master list to keep
        kept_sources=moc_routines.sources_in_tile(tiles[i],order,prior250.sra,prior250.sdec)
        #create indices for posterior (i.e. inlcude backgrounds and sigma_conf)
        ind_mast=np.array(kept_sources+[False]+kept_sources+[False]+kept_sources+[False]+[False,False,False])

        print sum(ind_mast),len(ind_mast),sum(ind_tmp),len(ind_tmp),tmp_prior250.nsrc,lower.size,upper.size
        stan_fit_master[:,:,ind_mast]=lower+(upper-lower)*tmp_posterior.stan_fit[:,:,ind_tmp]
    with open(output_folder+'master_posterior.pkl', 'wb') as f:
        pickle.dump({'posterior':stan_fit_master},f)



def make_tile_catalogues(output_folder,Master_filename,chains=4,iters=750):
    import pickle
    import dill
    from xidplus import moc_routines, catalogue
    with open(output_folder+Master_filename, "rb") as f:
        Master = pickle.load(f)

    tiles=Master['tiles']
    order=Master['order']
    prior250=Master['psw']
    for i in range(0,len(tiles)):
        print 'On tile '+str(i)+' out of '+str(len(tiles))
        infile=output_folder+'Tile_'+str(tiles[i])+'_'+str(order)+'.pkl'
        with open(infile, "rb") as f:
            obj = pickle.load(f)
        tmp_prior250=obj['psw']
        tmp_prior350=obj['pmw']
        tmp_prior500=obj['plw']
        tmp_posterior=obj['posterior']

        hdulist=catalogue.create_XIDp_SPIREcat_nocov(tmp_posterior,tmp_prior250,tmp_prior350,tmp_prior500)
        #work out what sources in tile to keep
        kept_sources=moc_routines.sources_in_tile(tiles[i],order,tmp_prior250.sra,tmp_prior250.sdec)

        kept_sources=np.array(kept_sources)

        hdulist[1].data=hdulist[1].data[kept_sources]
        hdulist.writeto(output_folder+'Tile_'+str(tiles[i])+'_'+str(order)+'.fits')
        hdulist.close()




def match_samples(prior,posterior,prior_tile,c,master_posterior,master_Rhat,master_n_eff,thresh=0.1):

    ind_mast=[]
    #get co-ords of temp prior list
    c_prior = SkyCoord(ra=prior.sra*u.degree, dec=prior.sdec*u.degree, frame='icrs')
    #where where in temp prior list, our source is
    s=c.match_to_catalog_sky(c_prior)[0]
    chains,iters,nparam=posterior.stan_fit.shape
    #get correlation coeffiecient, and only take uper tri of matrix
    cov=np.triu(np.corrcoef(posterior.stan_fit[:,:,:-3].reshape((chains*iters,nparam-3)).T),0)
    #select those sources where correlation is above threshold
    index=np.abs(cov[:,s])>thresh
    #select where in prior list of the tile, those correlated sources are
    for i in posterior.ID[index[:,0]]:
        ind_mast.append(np.arange(0,prior_tile.nsrc)[prior_tile.ID == i][0])

    #check if those correlated sources have already been filled in in master posterior
    match_id=[]
    for i in ind_mast:
        match_id.append(np.isfinite(master_posterior[0,i,0]))

    icp_ind_2=np.argsort(posterior.stan_fit[:,:,s].reshape(chains*iters,1),axis=0)
    #for all those sources that have been filled in already, match them
    if sum(match_id) >0:

        icp_ind=np.argsort(master_posterior[:,prior_tile.ID == prior.ID[s],0],axis=0)
        #find the index of the nearest neighbours in posterior
        fitted_s=s == np.array(np.arange(0,prior.nsrc)[index[:,0]])[np.array(match_id)]
        #those sources not already in master list
        print fitted_s,np.array(np.arange(0,prior.nsrc)[index[:,0]])[np.array(match_id)],[not i for i in match_id],match_id,ind_mast,s,np.array(np.arange(0,prior.nsrc)[index[:,0]])
        mast_ind=[not i for i in match_id]+fitted_s
        master_posterior[icp_ind,np.array(ind_mast)[mast_ind],0]=posterior.stan_fit[:,:,np.array(np.arange(0,prior.nsrc)[index[:,0]])[mast_ind]].reshape(chains*iters,sum(mast_ind))[icp_ind].reshape(chains*iters,sum(mast_ind))
        master_posterior[icp_ind,np.array(ind_mast)[mast_ind],1]=posterior.stan_fit[:,:,-3].reshape(chains*iters)[icp_ind_2]
        master_posterior[icp_ind,np.array(ind_mast)[mast_ind],2]=posterior.stan_fit[:,:,-2].reshape(chains*iters)[icp_ind_2]


    else:
        master_posterior[:,prior_tile.ID == prior.ID[s],0]=posterior.stan_fit[:,:,s].reshape(chains*iters)[icp_ind_2]
        master_posterior[:,prior_tile.ID == prior.ID[s],1]=posterior.stan_fit[:,:,-3].reshape(chains*iters)[icp_ind_2]
        master_posterior[:,prior_tile.ID == prior.ID[s],2]=posterior.stan_fit[:,:,-2].reshape(chains*iters)[icp_ind_2]
    master_Rhat[prior_tile.ID == prior.ID[s],0]=posterior.Rhat[s]
    master_Rhat[prior_tile.ID == prior.ID[s],1]=posterior.Rhat[-3]
    master_Rhat[prior_tile.ID == prior.ID[s],2]=posterior.Rhat[-2]
    master_n_eff[prior_tile.ID == prior.ID[s],0]=posterior.n_eff[s]
    master_n_eff[prior_tile.ID == prior.ID[s],1]=posterior.n_eff[-3]
    master_n_eff[prior_tile.ID == prior.ID[s],2]=posterior.n_eff[-2]

    return master_posterior,master_Rhat,master_n_eff





