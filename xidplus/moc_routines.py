__author__ = 'pdh21'
from mocpy import MOC
from healpy import pixelfunc

import numpy as np
def get_HEALPix_pixels(order,ra,dec,unique=True):
    """
    :param order: the HEALPix resolution level
    :param prior: Prior XID+ class, with image and catalogue set
    :return: list of pixels that need to be fit
    """

    HPX_D2R=np.pi/180.0
    #Convert catalogue to polar co-ords in radians
    phi = ra*HPX_D2R
    theta = np.pi/2.0 - dec*HPX_D2R
    #calculate what pixel each object is in
    ipix = pixelfunc.ang2pix(2**order, theta, phi, nest=True)
    #return unique pixels (i.e. remove duplicates)
    if unique is True:
        return np.unique(ipix)
    else:
        return ipix

def get_fitting_region(order,pixel):
    """
    expand tile by half a pixel for fitting
    :param order:the HEALPix resolution level
    :param pixel:given HEALPix pixel that needs to be fit
    :return: HEALPix pixels that need to be fit
    """
    #define old and new order
    old_nside=2**order
    new_nside=2**(order+1)

    #get co-ord of main pixel
    theta,phi=pixelfunc.pix2ang(old_nside, pixel, nest=True)
    #define offsets such that main pixel is split into four sub pixels
    scale=pixelfunc.max_pixrad(old_nside)
    offset_theta=np.array([-0.25,0.0,0.25,0.0])*scale
    offset_phi=np.array([0.0,-0.25,0.0,0.25])*scale
    #convert co-ords to pixels at higher order
    pix_fit=pixelfunc.ang2pix(new_nside, theta+offset_theta, phi+offset_phi, nest=True)
    #get neighbouring pixels and remove duplicates
    moc_tile=MOC()
    print np.unique(pixelfunc.get_all_neighbours(new_nside, pix_fit,nest=True))
    moc_tile.add_pix_list(order+1,np.unique(pixelfunc.get_all_neighbours(new_nside, pix_fit,nest=True)), nest=True)
    return moc_tile


def create_MOC_from_map(good,wcs):
    x_pix,y_pix=np.meshgrid(np.arange(0,wcs._naxis1),np.arange(0,wcs._naxis2))
    ra,dec= wcs.wcs_pix2world(x_pix,y_pix,0)

    pixels=get_HEALPix_pixels(12,ra[good],dec[good])
    map_moc=MOC()
    map_moc.add_pix_list(12,pixels, nest=True)
    return map_moc

def create_MOC_from_cat(ra,dec):
    pixels=get_HEALPix_pixels(12,ra,dec)
    cat_moc=MOC()
    cat_moc.add_pix_list(12,pixels, nest=True)
    return cat_moc

def check_in_moc(ra,dec,moc,keep_inside=True):
    pixels_best_res = set()
    kept_rows = []
    for val in moc.best_res_pixels_iterator():
        pixels_best_res.add(val)
    pix=get_HEALPix_pixels(moc.max_order,ra,dec,unique=False)
    for ipix in pix:
        kept_rows.append((ipix in pixels_best_res) == keep_inside)
    return kept_rows