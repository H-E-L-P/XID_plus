import numpy as np
from astropy import wcs

from xidplus import moc_routines


class prior(object):
    def cut_down_map(self):
        """Cuts down prior class variables associated with the map data to the MOC assigned to the prior class: self.moc
        """
        wcs_temp = wcs.WCS(self.imhdu)
        ra, dec = wcs_temp.wcs_pix2world(self.sx_pix, self.sy_pix, 0)
        ind_map = np.array(moc_routines.check_in_moc(ra, dec, self.moc))
        # now cut down and flatten maps (default is to use all pixels, running segment will change the values below to pixels within segment)
        self.sx_pix = self.sx_pix[ind_map]
        self.sy_pix = self.sy_pix[ind_map]
        self.snim = self.snim[ind_map]
        self.sim = self.sim[ind_map]
        self.snpix = sum(ind_map)

    def cut_down_cat(self):
        """Cuts down prior class variables associated with the catalogue data to the MOC assigned to the prior class: self.moc
        """
        sgood = np.array(moc_routines.check_in_moc(self.sra, self.sdec, self.moc))

        self.sx = self.sx[sgood]
        self.sy = self.sy[sgood]
        self.sra = self.sra[sgood]
        self.sdec = self.sdec[sgood]
        self.nsrc = sum(sgood)
        self.ID = self.ID[sgood]
        if hasattr(self, 'nstack'):
            self.stack = self.stack[sgood]
            self.nstack = sum(self.stack)
        if hasattr(self, 'prior_flux_upper'):
            self.prior_flux_upper = self.prior_flux_upper[sgood]
        if hasattr(self, 'prior_flux_lower'):
            self.prior_flux_lower = self.prior_flux_lower[sgood]

    def cut_down_prior(self):
        """ Cuts down prior class variables to the MOC assigned to the prior class: self.moc
        """
        self.cut_down_map()
        self.cut_down_cat()

    def __init__(self, im, nim, imphdu, imhdu, moc=None):

        # ---for any bad pixels set map pixel to zero and uncertianty to 1----
        """Initiate prior class

        :param im: image map from fits file
        :param nim: noise map from fits file
        :param imphdu: Primary header associated with fits file
        :param imhdu: header associated with image map
        :param moc: (default=None) Multi-Order Coverage map of area being fit
        """
        bad = np.logical_or(np.logical_or
                            (np.invert(np.isfinite(im)),
                             np.invert(np.isfinite(nim))), (nim == 0))
        if (bad.sum() > 0):
            im[bad] = 0.
            nim[bad] = 1.
        self.imhdu = imhdu
        wcs_temp = wcs.WCS(self.imhdu)
        self.imphdu = imphdu
        self.imhdu = imhdu

        x_pix, y_pix = np.meshgrid(np.arange(0, wcs_temp._naxis1), np.arange(0, wcs_temp._naxis2))
        self.sx_pix = x_pix.flatten()
        self.sy_pix = y_pix.flatten()
        self.snim = nim.flatten()
        self.sim = im.flatten()
        self.snpix = self.sim.size
        if moc is not None:
            self.moc = moc
            self.cut_down_map()

    def prior_bkg(self, mu, sigma):
        """Add background prior. Assumes normal distribution with mean=mu and standard deviation=sigma

        :param mu: mean
        :param sigma: standard deviation
        """

        self.bkg = (mu, sigma)

    def prior_cat(self, ra, dec, prior_cat_file, flux_lower=None, flux_upper=None, ID=None, moc=None):
        """Input info for prior catalogue

        :param ra: Right ascension (JD2000) of sources
        :param dec: Declination (JD2000) of sources
        :param prior_cat_file: filename of catalogue
        :param flux_lower: lower limit of flux for each source
        :param flux_upper: upper limit of flux for each source
        :param ID: HELP_ID for each source
        :param moc: Multi-Order Coverage map
        """
        # get positions of sources in terms of pixels
        wcs_temp = wcs.WCS(self.imhdu)
        sx, sy = wcs_temp.wcs_world2pix(ra, dec, 0)
        if moc is None:
            cat_moc = moc_routines.create_MOC_from_cat(ra, dec)
        else:
            cat_moc = moc


        # Redefine prior list so it only contains sources in the map
        self.sx = sx
        self.sy = sy
        self.sra = ra
        self.sdec = dec
        self.nsrc = self.sra.size
        self.prior_cat = prior_cat_file
        if flux_lower is None:
            flux_lower = np.full((ra.size), 0.01)
            flux_upper = np.full((ra.size), 1000.0)
        self.prior_flux_lower = flux_lower
        self.prior_flux_upper = flux_upper

        if ID is None:
            ID = np.arange(1, ra.size + 1, dtype='int64')
        self.ID = ID

        self.moc = cat_moc
        self.stack = np.full(self.nsrc, False)
        self.cut_down_prior()

    def set_tile(self, moc):
        """ Update prior with new MOC and update appropriate variables
        :param moc: Multi-order Coverage map from pymoc
        """
        self.moc = self.moc.intersection(moc)
        self.cut_down_prior()

    def prior_cat_stack(self, ra, dec, prior_cat, flux_lower=None, flux_upper=None,ID=None):
        """Input info for prior catalogue of sources being stacked

        :param ra: Right ascension (JD2000) of sources
        :param dec: Declination (JD2000) of sources
        :param prior_cat: filename of catalogue
        :param ID: HELP_ID for each source
        """
        wcs_temp = wcs.WCS(self.imhdu)
        sx, sy = wcs_temp.wcs_world2pix(ra, dec, 0)


        # Redefine prior list so it only contains sources in the map

        # Redefine prior list so it only contains sources in the map
        self.sx = np.append(self.sx, sx)
        self.sy = np.append(self.sy, sy)
        self.sra = np.append(self.sra, ra)
        self.sdec = np.append(self.sdec, dec)
        self.nstack = ra.size
        self.nsrc = self.sra.size
        self.stack = np.append(self.stack, np.full((self.nstack), True))
        if ID is None:
            ID = np.arange(1, ra.size + 1, dtype='int64')
        self.ID = np.append(self.ID, ID)
        if flux_lower is None:
            flux_lower = np.full((ra.size), 0.01)
            flux_upper = np.full((ra.size), 1000.0)
            self.prior_flux_lower = np.append(self.prior_flux_lower,flux_lower)
            self.prior_flux_upper = np.append(self.prior_flux_upper,flux_upper)

        self.cut_down_prior()

    def set_prf(self, prf, pindx, pindy):
        """Add prf array and corresponding x and y scales (in terms of pixels in map)

        :param prf: n x n array, where n is an odd number, and the centre of the prf is at the centre of the array
        :param pindx: n array, pixel scale of prf array
        :param pindy: n array, pixel scale of prf array
        """

        self.prf = prf
        self.pindx = pindx
        self.pindy = pindy

    def get_pointing_matrix(self, bkg=True):
        """Calculate pointing matrix. If bkg = True, bkg is fitted to all pixels. If False, bkg only fitted to where prior sources contribute
        """
        from scipy import interpolate
        paxis1, paxis2 = self.prf.shape

        amat_row = np.array([], dtype=int)
        amat_col = np.array([], dtype=int)
        amat_data = np.array([])

        # ------Deal with PRF array----------
        centre = ((paxis1 - 1) / 2)
        # create pointing array
        for s in range(0, self.nsrc):

            # diff from centre of beam for each pixel in x
            dx = -np.rint(self.sx[s]).astype(long) + self.pindx[(paxis1 - 1.) / 2] + self.sx_pix
            # diff from centre of beam for each pixel in y
            dy = -np.rint(self.sy[s]).astype(long) + self.pindy[(paxis2 - 1.) / 2] + self.sy_pix
            # diff from each pixel in prf
            pindx = self.pindx + self.sx[s] - np.rint(self.sx[s]).astype(long)
            pindy = self.pindy + self.sy[s] - np.rint(self.sy[s]).astype(long)

            good = (dx >= 0) & (dx < self.pindx[paxis1 - 1]) & (dy >= 0) & (dy < self.pindy[paxis2 - 1])
            ngood = good.sum()
            bad = np.asarray(good) == False
            nbad = bad.sum()
            if ngood > 0.5 * self.pindx[-1] * self.pindy[-1]:
                ipx2, ipy2 = np.meshgrid(pindx, pindy)
                atemp = interpolate.griddata((ipx2.ravel(), ipy2.ravel()), self.prf.ravel(), (dx[good], dy[good]),
                                             method='nearest')
                amat_data = np.append(amat_data, atemp)
                amat_row = np.append(amat_row,
                                     np.arange(0, self.snpix, dtype=int)[good])  # what pixels the source contributes to
                amat_col = np.append(amat_col, np.full(ngood, s))  # what source we are on

        self.amat_data = amat_data
        self.amat_row = amat_row
        self.amat_col = amat_col

    def get_pointing_matrix_coo(self):
        """Get scipy coo version of pointing matrix. Useful for sparse matrix multiplication"""
        from scipy.sparse import coo_matrix
        self.A = coo_matrix((self.amat_data, (self.amat_row, self.amat_col)), shape=(self.snpix, self.nsrc))



    def upper_lim_map(self):
        """Update flux upper limit to |bkg|+2*sigma_bkg+max(D) where max(D) is maximum value of pixels the source contributes to"""
        self.prior_flux_upper = np.full((self.nsrc), 1000.0)
        for i in range(0, self.nsrc):
            ind = self.amat_col == i
            if ind.sum() > 0:
                self.prior_flux_upper[i] = np.max(self.sim[self.amat_row[ind]]) + (np.abs(self.bkg[0]) + 2 * self.bkg[1])

    def get_pointing_matrix_unknown_psf(self, bkg=True):
        """Calculate pointing matrix for unknown psf. If bkg = True, bkg is fitted to all pixels. If False, bkg only fitted to where prior sources contribute
        """
        from scipy import interpolate
        paxis1, paxis2 = self.prf.shape

        amat_row = np.array([], dtype=int)
        amat_col = np.array([], dtype=int)
        amat_data = np.array([])

        # ------Deal with PRF array----------
        centre = ((paxis1 - 1) / 2)
        # create pointing array
        for s in range(0, self.nsrc):

            # diff from centre of beam for each pixel in x
            dx =  self.sx_pix-self.sx[s]
            # diff from centre of beam for each pixel in y
            dy =  self.sy_pix -self.sy[s]
            # diff from each pixel in prf
            pindx = self.pindx - (paxis1 - 1.) / 2. + self.sx[s] - np.rint(self.sx[s]).astype(long)
            pindy = self.pindy + self.sy[s] - np.rint(self.sy[s]).astype(long)
            # diff from pixel centre
            px = self.sx[s] - np.rint(self.sx[s]).astype(long) + (paxis1 - 1.) / 2.
            py = self.sy[s] - np.rint(self.sy[s]).astype(long) + (paxis2 - 1.) / 2.

            dist=np.sqrt(dx**2+dy**2)
            good = dist < self.pindx[-1]/2.0
            ngood = good.sum()
            bad = np.asarray(good) == False
            nbad = bad.sum()
            if ngood > 0.5 * self.pindx[-1] * self.pindy[-1]:
                f = interpolate.interp1d(self.pindx[0:(paxis1 + 1.) / 2],np.arange((paxis1 + 1.) / 2.),kind='nearest')
                atemp=f(dist[good])
                amat_data = np.append(amat_data, atemp)
                amat_row = np.append(amat_row,
                                     np.arange(0, self.snpix, dtype=int)[good])  # what pixels the source contributes to
                amat_col = np.append(amat_col, np.full(ngood, s))  # what source we are on

        self.amat_data = amat_data
        self.amat_row = amat_row
        self.amat_col = amat_col