import numpy as np
from astropy import wcs

class prior(object):
    def __init__(self,im,nim,imphdu,imhdu):
        """class for SPIRE prior object. Initialise with map,uncertianty map and wcs"""
        #---for any bad pixels set map pixel to zero and uncertianty to 1----
        bad=np.logical_or(np.logical_or
                  (np.invert(np.isfinite(im)),
                   np.invert(np.isfinite(nim))),(nim == 0))
        if(bad.sum() >0):
            im[bad]=0.
            nim[bad]=1.
        self.im=im
        self.nim=nim
        self.imhdu=imhdu
        wcs_temp=wcs.WCS(self.imhdu)
        self.imphdu=imphdu
        self.imhdu=imhdu
        #add a boolean array 
        ind=np.empty_like(im,dtype=bool)
        ind[:]=True
        #get x and y pixel position for each position
        x_pix,y_pix=np.meshgrid(np.arange(0,wcs_temp._naxis1),np.arange(0,wcs_temp._naxis2))
        #now cut down and flatten maps (default is to use all pixels, running segment will change the values below to pixels within segment)
        self.sx_pix=x_pix[ind]
        self.sy_pix=y_pix[ind]
        self.snim=self.nim[ind]
        self.sim=self.im[ind]
        self.snpix=ind.sum()


    def prior_bkg(self,mu,sigma):
        """Add background prior ($\mu$) and uncertianty ($\sigma$). Assumes normal distribution"""
        self.bkg=(mu,sigma)

    def prior_cat(self,ra,dec,prior_cat_file,ID=None,good_index=None,flux=None):
        """Input info for prior catalogue. Requires ra, dec and filename of prior cat. Checks sources in the prior list are within the boundaries of the map,
        and converts RA and DEC to pixel positions"""
        #get positions of sources in terms of pixels
        wcs_temp=wcs.WCS(self.imhdu)
        sx,sy=wcs_temp.wcs_world2pix(ra,dec,0)
        #check if sources are within map
        if hasattr(self, 'tile'):
            sgood=(ra > self.tile[0,0]-self.buffer_size) & (ra < self.tile[0,2]+self.buffer_size) & (dec > self.tile[1,0]-self.buffer_size) & (dec < self.tile[1,2]+self.buffer_size)#
        else:
            sgood=(sx > 0) & (sx < wcs_temp._naxis1) & (sy > 0) & (sy < wcs_temp._naxis2)
        #Redefine prior list so it only contains sources in the map
        self.sx=sx[sgood]
        self.sy=sy[sgood]
        self.sra=ra[sgood]
        self.sdec=dec[sgood]
        self.nsrc=sgood.sum()
        self.prior_cat=prior_cat_file
        if ID != None:
            self.ID=ID[sgood]
        if good_index != None:
            return sgood 
        if flux !=None:
            self.sflux=flux[sgood]


    def prior_cat_stack(self,ra,dec,prior_cat,good_index=None):
        """Input info for prior catalogue of sources being stacked. Requires ra, dec and filename of prior cat. Checks sources in the prior list are within the boundaries of the map,
        and converts RA and DEC to pixel positions"""
        #get positions of sources in terms of pixels
        wcs_temp=wcs.WCS(self.imhdu)
        sx,sy=wcs_temp.wcs_world2pix(ra,dec,0)
        #check if sources are within map 
        sgood=(ra > self.tile[0,0]-self.buffer_size) & (ra < self.tile[0,2]+self.buffer_size) & (dec > self.tile[1,0]-self.buffer_size) & (dec < self.tile[1,2]+self.buffer_size)# & np.isfinite(im250[np.rint(sx250).astype(int),np.rint(sy250).astype(int)])#this gives boolean array for cat

                

        #Redefine prior list so it only contains sources in the tile being fitted
        self.stack_sx=sx[sgood]
        self.stack_sy=sy[sgood]
        self.stack_sra=ra[sgood]
        self.stack_sdec=dec[sgood]
        if hasattr(self, 'sx'):
            self.sx=np.append(self.sx,sx[sgood])
            self.sy=np.append(self.sy,sy[sgood])
            self.sra=np.append(self.sra,ra[sgood])
            self.sdec=np.append(self.sdec,dec[sgood])
            self.nsrc=self.nsrc+sgood.sum()
        else:
            self.sx=sx[sgood]
            self.sy=sy[sgood]
            self.sra=ra[sgood]
            self.sdec=dec[sgood]
            self.nsrc=sgood.sum()
        self.stack_nsrc=sgood.sum()
        if good_index != None:
            return sgood 
    
    def set_tile(self,tile,buffer_size):
        """Segment map to tile region described by tile and buffer_size"""
        #create polygon of tile (in format used by aplpy). Should be 2x4 array
        self.tile=tile
        #get vertices of polygon in terms of pixels
        wcs_temp=wcs.WCS(self.imhdu)

        tile_x,tile_y=wcs_temp.wcs_world2pix(tile[0,:],tile[1,:],0)

        x_pix,y_pix=np.meshgrid(np.arange(0,wcs_temp._naxis1),np.arange(0,wcs_temp._naxis2))

        npix=(x_pix < np.max(tile_x)) & (y_pix < np.max(tile_y)) & (y_pix >= np.min(tile_y)) & (x_pix >= np.min(tile_x))

        #now cut down and flatten maps
        self.sx_pix=x_pix[npix]
        self.sy_pix=y_pix[npix]
        self.snim=self.nim[npix]
        self.sim=self.im[npix]
        self.snpix=npix.sum()

        
        #store buffer size
        self.buffer_size=buffer_size


    def set_prf(self,prf,pindx,pindy):
        """Add prf array and corresponding x and y scales (in terms of pixels in map). \n Array should be an n x n array, where n is an odd number, and the centre of the prf is at the centre of the array"""
        self.prf=prf
        self.pindx=pindx
        self.pindy=pindy
        

    def get_pointing_matrix(self, bkg=True):
        """get the pointing matrix. If bkg = True, bkg is fitted to all pixels. If False, bkg only fitted to where prior sources contribute"""
        from scipy import interpolate        
        paxis1,paxis2=self.prf.shape

        amat_row=np.array([])
        amat_col=np.array([])
        amat_data=np.array([])
        
        #------Deal with PRF array----------
        centre=((paxis1-1)/2)
        #create pointing array
        for s in range(0,self.nsrc):



            #diff from centre of beam for each pixel in x
            dx = -np.rint(self.sx[s]).astype(long)+self.pindx[(paxis1-1.)/2]+self.sx_pix
            #diff from centre of beam for each pixel in y
            dy = -np.rint(self.sy[s]).astype(long)+self.pindy[(paxis2-1.)/2]+self.sy_pix
            #diff from each pixel in prf
            pindx=self.pindx+self.sx[s]-np.rint(self.sx[s]).astype(long)
            pindy=self.pindy+self.sy[s]-np.rint(self.sy[s]).astype(long)
            #diff from pixel centre
            px=self.sx[s]-np.rint(self.sx[s]).astype(long)+(paxis1-1.)/2.
            py=self.sy[s]-np.rint(self.sy[s]).astype(long)+(paxis2-1.)/2.
        
            good = (dx >= 0) & (dx < self.pindx[paxis1-1]) & (dy >= 0) & (dy < self.pindy[paxis2-1])
            ngood = good.sum()
            bad = np.asarray(good)==False
            nbad=bad.sum()
            if ngood > 0.5*self.pindx[-1]*self.pindy[-1]:
                ipx2,ipy2=np.meshgrid(pindx,pindy)
                atemp=interpolate.griddata((ipx2.ravel(),ipy2.ravel()),self.prf.ravel(), (dx[good],dy[good]), method='nearest')
                amat_data=np.append(amat_data,atemp)
                amat_row=np.append(amat_row,np.arange(0,self.snpix,dtype=int)[good])#what pixels the source contributes to
                amat_col=np.append(amat_col,np.full(ngood,s))#what source we are on
        

            self.amat_data=amat_data
            self.amat_row=amat_row
            self.amat_col=amat_col


    def get_pointing_matrix_coo(self):
        """Get scipy coo version of pointing matrix. Useful for sparse matrix multiplication"""
        from scipy.sparse import coo_matrix
        self.A=coo_matrix((self.amat_data, (self.amat_row, self.amat_col)), shape=(self.snpix, self.nsrc))

    def cut_map_to_prior(self):
        """If only interested in fitting around regions of prior objects, run this function to cut down amount of data being fitted to."""
        ind=np.unique(self.amat_row).astype(int)
        #Remove pixels from class that are not being fitted (i.e. which don't appear in the pointing matrix)
        #now cut down and flatten maps
        self.sx_pix=self.sx_pix[ind]
        self.sy_pix=self.sy_pix[ind]
        self.snim=self.snim[ind]
        self.sim=self.sim[ind]
        self.snpix=ind.size
        self.get_pointing_matrix()
