__author__ = 'pdh21'
from pymoc import MOC
from healpy import pixelfunc
from pymoc.util import catalog
import healpy as hp

import numpy as np

def get_HEALPix_pixels(order,ra,dec,unique=True):


    """Work out what HEALPix a source is in

    :param order: order of HEALPix
    :param ra: Right Ascension
    :param dec: Declination
    :param unique: if unique is true, removes duplicate pixels
    :return: list of HEALPix pixels :rtype:
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
    """Expand tile by quarter of a pixel for fitting

    
    :param order: the HEALPix resolution level
    :param pixel: given HEALPix pixel that needs to be fit
    :return: HEALPix pixels that need to be fit
    """

    #define old and new order
    old_nside=2**order
    new_nside=2**(order+2)

    #get co-ord of main pixel
    theta,phi=pixelfunc.pix2ang(old_nside, pixel, nest=True)
    #define offsets such that main pixel is split into four sub pixels
    scale=pixelfunc.max_pixrad(old_nside)
    offset_theta=np.array([-0.125,0.0,0.125,0.0])*scale
    offset_phi=np.array([0.0,-0.125,0.0,0.125])*scale
    #convert co-ords to pixels at higher order
    pix_fit=pixelfunc.ang2pix(new_nside, theta+offset_theta, phi+offset_phi, nest=True)
    #get neighbouring pixels and remove duplicates
    moc_tile=MOC()
    pixels=np.unique(pixelfunc.get_all_neighbours(new_nside, pix_fit,nest=True))
    moc_tile.add(order+2,np.unique(pixelfunc.get_all_neighbours(new_nside, pixels,nest=True)))
    return moc_tile


def create_MOC_from_map(good,wcs):
    """Generate MOC from map


    :param good: boolean array associated with map
    :param wcs: wcs information
    :return: MOC :rtype: pymoc.MOC
    """
    x_pix,y_pix=np.meshgrid(np.arange(0,wcs._naxis1),np.arange(0,wcs._naxis2))
    ra,dec= wcs.wcs_pix2world(x_pix,y_pix,0)

    pixels=get_HEALPix_pixels(15,ra[good],dec[good])
    map_moc=MOC()
    map_moc.add(15,pixels)
    return map_moc

def create_MOC_from_cat(ra,dec):
    """Generate MOC from catalogue

    :param ra: Right ascension of sources
    :param dec: Declination of sources
    :return: MOC :rtype: pymoc.MOC
    """
    pixels=get_HEALPix_pixels(11,ra,dec)
    cat_moc=MOC()
    cat_moc.add(11,pixels)
    return cat_moc

def coords_to_hpidx(ra, dec, order):
    """Convert coordinates to HEALPix indexes
    Given to list of right ascension and declination, this function computes
    the HEALPix index (in nested scheme) at each position, at the given order.
    Parameters
    ----------
    ra: array or list of floats
        The right ascensions of the sources.
    dec: array or list of floats
        The declinations of the sources.
    order: int
        HEALPix order.
    Returns
    -------
    array of int
        The HEALPix index at each position.
    """
    ra, dec = np.array(ra), np.array(dec)

    theta = 0.5 * np.pi - np.radians(dec)
    phi = np.radians(ra)
    healpix_idx = hp.ang2pix(2**order, theta, phi, nest=True)

    return healpix_idx

def check_in_moc(ra, dec, moc):
    """Find source position in a MOC
    Given a list of positions and a Multi Order Coverage (MOC) map, this
    function return a boolean mask with True for sources that fall inside the
    MOC and False elsewhere.
    Parameters
    ----------
    ra: array or list of floats
        The right ascensions of the sources.
    dec: array or list of floats
        The declinations of the sources.
    moc: pymoc.MOC
        The MOC read by pymoc
    Returns
    -------
    array of booleans
        The boolean mask with True for sources that fall inside the MOC.
    """
    source_healpix_cells = coords_to_hpidx(
        np.array(ra), np.array(dec), moc.order
    )

    # Array of all the HEALpix cell ids of the MOC at its maximum order.
    moc_healpix_cells = np.array(list(moc.flattened()))

    # We look for sources that are in the MOC and return the mask
    return np.in1d(source_healpix_cells, moc_healpix_cells)

def check_in_moc_pdh(ra,dec,moc):
    """Check whether a source is in MOC or not

    :param ra: Right Ascension
    :param dec: Declination
    :param moc: MOC
    :return: boolean array expressing whether in MOC or not :rtype: boolean array
    """
    kept_rows = []
    pix=get_HEALPix_pixels(moc.order,ra,dec,unique=False)
    for ipix in pix:
        kept_rows.append(moc.contains(moc.order,ipix, include_smaller=True))
    return kept_rows



def sources_in_tile(pixel,order,ra,dec):
    """Check which sources are in HEALPix pixel

    :param pixel: HEALPix pixel
    :param order: order of HEALPix pixel
    :param ra: Right Ascension
    :param dec: Declination
    :return: boolean array expressing whether in MOC or not :rtype:boolean array
    """
    moc_pix=MOC()
    moc_pix.add(order,pixel)
    kept_sources=check_in_moc(ra,dec,moc_pix)
    return kept_sources

def tile_in_tile(order_small,tile_small,order_large):
    """Routine to find our what larger tile to load data from when fitting smaller tiles.
     Returns larger tile no. Useful for when fitting segmented maps on HPC using a hierarchical segmentation scheme

    :param order_small: order of smaller tile
    :param tile_small:  tile no. of smaller tile
    :param order_large: order of larger tiling scheme
    :return: pixel number of larger tile
    """
    theta, phi =pixelfunc.pix2ang(2**order_small, tile_small, nest=True)
    ipix = pixelfunc.ang2pix(2**order_large, theta, phi, nest=True)
    return ipix
