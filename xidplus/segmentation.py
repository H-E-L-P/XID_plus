import numpy as np


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







