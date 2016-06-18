__author__ = 'pdh21'
from astropy.io import fits

import numpy as np
from astropy.io import fits


def ymod_map(prior,flux):
    """Create replicated model map (no noise or background) i.e. A*f

    :param prior: prior class
    :param flux: flux vector
    :return: map array, in same format as prior.sim
    """
    from scipy.sparse import coo_matrix

    f=coo_matrix((flux, (range(0,prior.nsrc),np.zeros(prior.nsrc))), shape=(prior.nsrc, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc))
    rmap_temp=(A*f)
    #pred_map=np.empty_like(prior.im)
    #pred_map[:,:]=0.0
    #pred_map[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)#+np.random.randn(prior.snpix)*prior.snim

    return np.asarray(rmap_temp.todense())


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

def make_fits_image(prior,pixel_values):
    """
    :param prior: prior class for XID+
    :param pixel_values:this is the pixel values returned from ymod_map
    :return: fits hdulist
    """
    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    data=np.full((y_range,x_range),np.nan)
    data[prior.sy_pix-np.min(prior.sy_pix)-1,prior.sx_pix-np.min(prior.sx_pix)-1]=pixel_values
    hdulist = fits.HDUList([fits.PrimaryHDU(header=prior.imphdu),fits.ImageHDU(data=data,header=prior.imhdu)])
    hdulist[1].header['CRPIX1']=hdulist[1].header['CRPIX1']-np.min(prior.sx_pix)-1
    hdulist[1].header['CRPIX2']=hdulist[1].header['CRPIX2']-np.min(prior.sy_pix)-1

    return hdulist