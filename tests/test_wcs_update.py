import unittest
import xidplus
from astropy.io import ascii, fits
from astropy import wcs


import numpy as np
import xidplus
from xidplus import moc_routines

class wcs_update(unittest.TestCase):
    def setUp(self):
        xidplus.__path__[0]
        # Folder containing maps
        imfolder = xidplus.__path__[0] + '/../test_files/'

        pswfits = imfolder + 'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PSW_hipe.fits.gz'  # SPIRE 250 map
        pmwfits = imfolder + 'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PMW_hipe.fits.gz'  # SPIRE 350 map
        plwfits = imfolder + 'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PLW_hipe.fits.gz'  # SPIRE 500 map

        # Folder containing prior input catalogue
        catfolder = xidplus.__path__[0] + '/../test_files/'
        # prior catalogue
        prior_cat = 'lacey_07012015_MillGas.ALLVOLS_cat_PSW_COSMOS_test.fits'

        # output folder
        output_folder = './'

        # -----250-------------
        hdulist = fits.open(pswfits)
        im250phdu = hdulist[0].header
        im250hdu = hdulist[1].header

        im250 = hdulist[1].data * 1.0E3  # convert to mJy
        nim250 = hdulist[2].data * 1.0E3  # convert to mJy
        w_250 = wcs.WCS(hdulist[1].header)
        pixsize250 = 3600.0 * w_250.wcs.cd[1, 1]  # pixel size (in arcseconds)
        hdulist.close()
        # -----350-------------
        hdulist = fits.open(pmwfits)
        im350phdu = hdulist[0].header
        im350hdu = hdulist[1].header

        im350 = hdulist[1].data * 1.0E3  # convert to mJy
        nim350 = hdulist[2].data * 1.0E3  # convert to mJy
        w_350 = wcs.WCS(hdulist[1].header)
        pixsize350 = 3600.0 * w_350.wcs.cd[1, 1]  # pixel size (in arcseconds)
        hdulist.close()
        # -----500-------------
        hdulist = fits.open(plwfits)
        im500phdu = hdulist[0].header
        im500hdu = hdulist[1].header
        im500 = hdulist[1].data * 1.0E3  # convert to mJy
        nim500 = hdulist[2].data * 1.0E3  # convert to mJy
        w_500 = wcs.WCS(hdulist[1].header)
        pixsize500 = 3600.0 * w_500.wcs.cd[1, 1]  # pixel size (in arcseconds)
        hdulist.close()

        hdulist = fits.open(catfolder + prior_cat)
        fcat = hdulist[1].data
        hdulist.close()
        inra = fcat['RA']
        indec = fcat['DEC']
        # select only sources with 100micron flux greater than 50 microJy
        sgood = fcat['S100'] > 0.050
        inra = inra[sgood]
        indec = indec[sgood]

        from astropy.coordinates import SkyCoord
        from astropy import units as u
        c = SkyCoord(ra=[150.74] * u.degree, dec=[2.03] * u.degree)
        import pymoc
        moc = pymoc.util.catalog.catalog_to_moc(c, 100, 15)

        # ---prior250--------
        prior250 = xidplus.prior(im250, nim250, im250phdu, im250hdu,
                                 moc=moc)  # Initialise with map, uncertianty map, wcs info and primary header
        prior250.prior_cat(inra, indec, prior_cat)  # Set input catalogue
        prior250.prior_bkg(-5.0, 5)  # Set prior on background (assumes Gaussian pdf with mu and sigma)
        # ---prior350--------
        prior350 = xidplus.prior(im350, nim350, im350phdu, im350hdu, moc=moc)
        prior350.prior_cat(inra, indec, prior_cat)
        prior350.prior_bkg(-5.0, 5)

        # ---prior500--------
        prior500 = xidplus.prior(im500, nim500, im500phdu, im500hdu, moc=moc)
        prior500.prior_cat(inra, indec, prior_cat)
        prior500.prior_bkg(-5.0, 5)

        self.priors=[prior250,prior350,prior500]
    def test_original_makefits(self):
        fit_image=xidplus.postmaps.make_fits_image(self.priors[0],self.priors[0].sim)
        xx,yy=np.meshgrid(np.arange(0,fit_image[1].data.shape[0]),np.arange(0,fit_image[1].data.shape[1]))
        print(xx.flatten().shape,self.priors[0].sx_pix.shape,self.priors[0].sy_pix.shape)

        self.assertEqual(xx.flatten().shape,self.priors[0].sy_pix.shape)
        self.assertEqual(yy.flatten().shape,self.priors[0].sy_pix.shape)






if __name__ == '__main__':
    unittest.main()
