import numpy as np
from astropy import wcs
from xidplus import moc_routines


class prior(object):
    def __init__(self,im,nim,imphdu,imhdu,moc=None):
        """class for SPIRE prior object. Initialise with map,uncertianty map and wcs"""
        #---for any bad pixels set map pixel to zero and uncertianty to 1----
        bad=np.logical_or(np.logical_or
                  (np.invert(np.isfinite(im)),
                   np.invert(np.isfinite(nim))),(nim == 0))
        if(bad.sum() >0):
            im[bad]=0.
            nim[bad]=1.
        self.imhdu=imhdu
        wcs_temp=wcs.WCS(self.imhdu)
        self.imphdu=imphdu
        self.imhdu=imhdu

        #if moc is None:
        #    self.moc=moc_routines.create_MOC_from_map(np.logical_not(bad),wcs_temp)
        #else:
        #    self.moc=moc

        x_pix,y_pix=np.meshgrid(np.arange(0,wcs_temp._naxis1),np.arange(0,wcs_temp._naxis2))
        self.sx_pix=x_pix.flatten()
        self.sy_pix=y_pix.flatten()
        self.snim=nim.flatten()
        self.sim=im.flatten()
        self.snpix=self.sim.size

    def cut_down_prior(self):
        wcs_temp=wcs.WCS(self.imhdu)
        ra,dec= wcs_temp.wcs_pix2world(self.sx_pix,self.sy_pix,0)
        ind_map=np.array(moc_routines.check_in_moc(ra,dec,self.moc,keep_inside=True))
        #now cut down and flatten maps (default is to use all pixels, running segment will change the values below to pixels within segment)
        self.sx_pix=self.sx_pix[ind_map]
        self.sy_pix=self.sy_pix[ind_map]
        self.snim=self.snim[ind_map]
        self.sim=self.sim[ind_map]
        self.snpix=sum(ind_map)



        sgood=np.array(moc_routines.check_in_moc(self.sra,self.sdec,self.moc,keep_inside=True))

        self.sx=self.sx[sgood]
        self.sy=self.sy[sgood]
        self.sra=self.sra[sgood]
        self.sdec=self.sdec[sgood]
        self.nsrc=sum(sgood)
        self.ID=self.ID[sgood]



    def prior_bkg(self,mu,sigma):
        """Add background prior ($\mu$) and uncertianty ($\sigma$). Assumes normal distribution"""
        self.bkg=(mu,sigma)

    def prior_cat(self,ra,dec,prior_cat_file,ID=None,moc=None):
        """Input info for prior catalogue. Requires ra, dec and filename of prior cat. Checks sources in the prior list are within the boundaries of the map,
        and converts RA and DEC to pixel positions"""
        #get positions of sources in terms of pixels
        wcs_temp=wcs.WCS(self.imhdu)
        sx,sy=wcs_temp.wcs_world2pix(ra,dec,0)
        if moc is None:
            cat_moc=moc_routines.create_MOC_from_cat(ra,dec)
        else:
            cat_moc=moc


        #Redefine prior list so it only contains sources in the map
        self.sx=sx
        self.sy=sy
        self.sra=ra
        self.sdec=dec
        self.nsrc=self.sra.size
        self.prior_cat=prior_cat_file
        if ID is None:
            ID=np.arange(1,ra.size+1,dtype='int64')
        self.ID=ID


        self.moc=cat_moc
        self.cut_down_prior()

    def set_tile(self,moc):
        self.moc=self.moc.intersection(moc)
        self.cut_down_prior()

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


    def set_prf(self,prf,pindx,pindy):
        """Add prf array and corresponding x and y scales (in terms of pixels in map). \n Array should be an n x n array, where n is an odd number, and the centre of the prf is at the centre of the array"""
        self.prf=prf
        self.pindx=pindx
        self.pindy=pindy
        

    def get_pointing_matrix(self, bkg=True):
        """get the pointing matrix. If bkg = True, bkg is fitted to all pixels. If False, bkg only fitted to where prior sources contribute"""
        from scipy import interpolate        
        paxis1,paxis2=self.prf.shape

        amat_row=np.array([],dtype=int)
        amat_col=np.array([],dtype=int)
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

    def upper_lim_map(self):
        self.prior_flux_upper=np.full((self.nsrc), 3.0)
        for i in range(0,self.nsrc):
            ind=self.amat_col == i
            if ind.sum() >0:
                self.prior_flux_upper[i]=np.max(self.sim[self.amat_row[ind]])-(self.bkg[0]-2*self.bkg[1])

    def upper_lim_flux(self,prior_flux_upper):
        """Set flux lower limit (in log10)"""
        self.prior_flux_upper=np.full((self.nsrc),prior_flux_upper)
    def lower_lim_flux(self,prior_flux_lower):
        """Set flux lower limit (in log10)"""
        self.prior_flux_lower=np.full((self.nsrc),prior_flux_lower)

    def get_pointing_matrix_map(self, bkg=True):
        """get the pointing matrix. If bkg = True, bkg is fitted to all pixels. If False, bkg only fitted to where prior sources contribute"""
        from scipy import interpolate
        paxis1,paxis2=self.prf.shape

        amat_row=np.array([],dtype=int)
        amat_col=np.array([],dtype=int)
        amat_data=np.array([])

        #------Deal with PRF array----------
        centre=((paxis1-1)/2)
        #create pointing array
        for s in range(0,self.snpix):



            #diff from centre of beam for each pixel in x
            dx = -np.rint(self.sx_pix[s]).astype(long)+self.pindx[(paxis1-1.)/2]+self.sx_pix
            #diff from centre of beam for each pixel in y
            dy = -np.rint(self.sy_pix[s]).astype(long)+self.pindy[(paxis2-1.)/2]+self.sy_pix
            #diff from each pixel in prf
            pindx=self.pindx+self.sx_pix[s]-np.rint(self.sx_pix[s]).astype(long)
            pindy=self.pindy+self.sy_pix[s]-np.rint(self.sy_pix[s]).astype(long)
            #diff from pixel centre
            px=self.sx_pix[s]-np.rint(self.sx_pix[s]).astype(long)+(paxis1-1.)/2.
            py=self.sy_pix[s]-np.rint(self.sy_pix[s]).astype(long)+(paxis2-1.)/2.

            good = (dx >= 0) & (dx < self.pindx[paxis1-1]) & (dy >= 0) & (dy < self.pindy[paxis2-1])
            ngood = good.sum()
            bad = np.asarray(good)==False
            nbad=bad.sum()
            if ngood > 0:#0.5*self.pindx[-1]*self.pindy[-1]:
                ipx2,ipy2=np.meshgrid(pindx,pindy)
                atemp=interpolate.griddata((ipx2.ravel(),ipy2.ravel()),self.prf.ravel(), (dx[good],dy[good]), method='nearest')
                amat_data=np.append(amat_data,atemp)
                amat_row=np.append(amat_row,np.arange(0,self.snpix,dtype=int)[good])#what pixels the source contributes to
                amat_col=np.append(amat_col,np.full(ngood,s))#what source we are on

        ind=amat_row <= amat_col
        self.amat_data_map=amat_data#[ind]
        self.amat_row_map=amat_row#[ind]
        self.amat_col_map=amat_col#[ind]

    def conf_noise(self):
        import confusion_noise as conf
        (self.Row_sig_conf,self.Col_sig_conf,self.Val_sig_conf,self.n_sig_conf)=conf.select_confusion_cov_max_pixels(4,self)