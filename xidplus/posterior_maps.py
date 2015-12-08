__author__ = 'pdh21'
import numpy as np

def ymod_map(prior250,prior350,prior500,posterior_sample):
    from scipy.sparse import coo_matrix

    f=coo_matrix((posterior_sample[0:prior250.nsrc], (range(0,prior250.nsrc),np.zeros(prior250.nsrc))), shape=(prior250.nsrc, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc))
    rmap_temp=(A*f)
    pred_map=np.empty_like(prior.im)
    pred_map[:,:]=0.0
    pred_map[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)#+np.random.randn(prior.snpix)*prior.snim

    return pred_map,np.asarray(rmap_temp.todense())


def yrep_map(prior,fvec,conf_noise):
    from scipy.sparse import coo_matrix


    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    f=coo_matrix((fvec, (range(0,prior.nsrc),np.zeros(prior.nsrc))), shape=(prior.nsrc, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc))
    rmap_temp=(A*f)
    pred_map=np.empty_like(prior.im)
    pred_map[:,:]=0.0
    pred_map[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)+np.random.randn(prior.snpix)*np.sqrt(np.square(prior.snim)+np.square(conf_noise))

    return pred_map,np.asarray(rmap_temp.todense())+np.random.randn(prior.snpix)*np.sqrt(np.square(prior.snim)+np.square(conf_noise))
