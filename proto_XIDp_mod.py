
# coding: utf-8

## XID high level code

# import modules

# In[1]:

import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs



def lstdrv_initsolveSP(sx,sy,prf,
                       sig_map,noisy_map,x_pix,y_pix):
                       #,bl,bu,psrc,psig,pmat,amat2,sn_map):
    from scipy import interpolate
    from scipy.sparse import coo_matrix

    snsrc=sx.size
    paxis1,paxis2=prf.shape
    #cut down map to sources being fitted
    #first get range around sources
    mux=np.max(sx)
    mlx=np.min(sx)
    muy=np.max(sy)
    mly=np.min(sy)
    npix=(x_pix < mux+20) & (y_pix < muy+20) & (y_pix >= mly-20) & (x_pix >= mlx-20)
    snpix=npix.sum()
    #now cut down and flatten maps
    sx_pix=x_pix[npix]
    sy_pix=y_pix[npix]
    snoisy_map=noisy_map[npix]
    ssig_map=sig_map[npix]
    amat_row=np.array([])
    amat_col=np.array([])
    amat_data=np.array([])

    #create pointing array
    for s in range(0,snsrc):



        #diff from centre of beam for each pixel in x
        dx = -np.rint(sx[s]).astype(long)+(paxis1-1.)/2.+sx_pix
        #diff from centre of beam for each pixel in y
        dy = -np.rint(sy[s]).astype(long)+(paxis2-1.)/2.+sy_pix
        #diff from each pixel in prf
        pindx=range(0,paxis1)+sx[s]-np.rint(sx[s]).astype(long)
        pindy=range(0,paxis2)+sy[s]-np.rint(sy[s]).astype(long)
        
        #diff from pixel centre
        px=sx[s]-np.rint(sx[s]).astype(long)+(paxis1-1.)/2.
        py=sy[s]-np.rint(sy[s]).astype(long)+(paxis2-1.)/2.
        
        good = (dx >= 0) & (dx < paxis1) & (dy >= 0) & (dy < paxis2)
        ngood = good.sum()
        bad = np.asarray(good)==False
        nbad=bad.sum()
        if ngood > 0.5*prf.array.size:
            ipx2,ipy2=np.meshgrid(pindx,pindy)
            nprf=interpolate.Rbf(ipx2.ravel(),ipy2.ravel(),prf.array.ravel(),function='cubic')
            atemp=np.empty((ngood))
            for i in range(0,ngood):
                atemp[i]=nprf(dx[good][i],dy[good][i])
            amat_data=np.append(amat_data,atemp)
            amat_row=np.append(amat_row,np.arange(0,snpix,dtype=long)[good])#what pixels the source contributes to
            amat_col=np.append(amat_col,np.full(ngood,s))#what source we are on

    #Add background contribution to pointing matrix
    amat_data=np.append(amat_data,np.full(snpix,1))
    amat_row=np.append(amat_row,np.arange(0,snpix,dtype=int))
    amat_col=np.append(amat_col,np.full(snpix,s+1))
    A=coo_matrix((amat_data, (amat_row, amat_col)), shape=(snpix, snsrc+1))
    return amat_data,amat_row,amat_col,A,sx_pix,sy_pix,snoisy_map,ssig_map,snsrc,snpix

    

    

    


# In[14]:

def convergence_stats(chain):
    #function to calculate the between and within-sequence variance,
    #marginal posterior variance, and R
    #for one parameter, as described in DAT,sec 11.4
    #(function will split each chain into two)
    #chain is a n,m array, n=number of iterations,m=number of chains
    #chain should not include warmup
    #will return B,W,var_psi_y,R
    n,m=chain.shape
    n_2=n/2.0
    psi_j=np.empty((2*m))
    s2_j=np.empty((2*m))
    for j in range(0,m):
        psi_j[j]=np.mean(chain[0:n/2.0,j])
        psi_j[j+m]=np.mean(chain[n/2.0:,j])
        #print np.power(chain[0:n/2.0,j]-psi_j[j],2)
        #print np.power(chain[n/2.0:,j]-psi_j[j+m],2)
        s2_j[j]=(1.0/((n/2.0)-1))*np.sum(np.power(chain[0:n/2.0,j]-psi_j[j],2))
        s2_j[j+m]=(1.0/((n/2.0)-1))*np.sum(np.power(chain[n/2.0:,j]-psi_j[j+m],2))

    psi=np.mean(psi_j)
    B=((n/2.0)/(2.0*m-1))*np.sum(np.power(psi_j-psi,2))
    W=np.mean(s2_j)
    var_psi_y=(((n_2-1)/n_2)*W)
    R=np.power(var_psi_y/W,0.5)
    return B,W,var_psi_y,R


# In[15]:

# define a function to get percentile for a particular parameter
def quantileGet(q,param):
    #q is quantile
    #param is array (nsamples,nparameters)
    # make a list to store the quantiles
    quants = []
 
    # for every predicted value
    for i in range(param.shape[1]):
        # make a vector to store the predictions from each chain
        val = []
 
        # next go down the rows and store the values
        for j in range(param.shape[0]):
            val.append(param[j,i])
 
        # return the quantile for the predictions.
        quants.append(np.percentile(val, q))
 
    return quants


# In[16]:

def lstdrv_stan(amat_data,amat_row,amat_col,snoisy_map,ssig_map,snsrc,snpix,bkg,sig_bkg,chains=4,iter=1000):
    #
    import pystan
    import pickle

    # define function to initialise flux values to one
    def initfun():
        return dict(src_f=np.ones(snsrc))
    #input data into a dictionary

    XID_data={'npix':snpix,
          'nsrc':snsrc,
          'nnz':amat_data.size,
          'db':ssig_map,
          'sigma':snoisy_map,
          'bkg_prior':bkg,
          'bkg_prior_sig':sig_bkg,
          'Val':amat_data,
          'Row': amat_row.astype(long),
          'Col': amat_col.astype(long)}
    
    #see if model has already been compiled. If not, compile and save it
    import os
    model_file="./XID+_basic.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)
            fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file='XIDfit.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    #run pystan with dictionary of data
    #fit=pystan.stan(file='XIDfit.stan',data=XID_data,iter=iter,chains=chains)#,init=initfun)
    #extract fit
    fit_data=fit.extract(permuted=False, inc_warmup=False)
    #return fit data
    return fit_data,chains,iter

def lstdrv_stan_highz(amat_data,amat_row,amat_col,snoisy_map,ssig_map,n_src,n_src_z,snpix,bkg,sig_bkg,chains=4,iter=1000):
    #
    import pystan
    import pickle

    # define function to initialise flux values to one
    def initfun():
        return dict(src_f=np.ones(snsrc))
    #input data into a dictionary

    XID_data={'npix':snpix,
          'nsrc':n_src+n_src_z,
          'nsrc_z':n_src_z,
          'nnz':amat_data.size,
          'db':ssig_map,
          'sigma':snoisy_map,
          'bkg_prior':bkg,
          'bkg_prior_sig':sig_bkg,
          'Val':amat_data,
          'Row': amat_row.astype(long),
          'Col': amat_col.astype(long)}
    
    #see if model has already been compiled. If not, compile and save it
    import os
    model_file="./XID+highz.pkl"
    try:
       with open(model_file,'rb') as f:
            # using the same model as before
            print("%s found. Reusing" % model_file)
            sm = pickle.load(f)
            fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    except IOError as e:
        print("%s not found. Compiling" % model_file)
        sm = pystan.StanModel(file='XID+highz.stan')
        # save it to the file 'model.pkl' for later use
        with open(model_file, 'wb') as f:
            pickle.dump(sm, f)
        fit = sm.sampling(data=XID_data,iter=iter,chains=chains)
    #run pystan with dictionary of data
    #fit=pystan.stan(file='XIDfit.stan',data=XID_data,iter=iter,chains=chains)#,init=initfun)
    #extract fit
    fit_data=fit.extract(permuted=False, inc_warmup=False)
    #return fit data
    return fit_data,chains,iter


# In[17]:

def lstdrv_solvefluxes(sx250,sy250,sx350,sy350,sx500,sy500#pixel positions of sources
                   ,prf250,prf350,prf500,#fwhm of beams in pixels
                   im250,im350,im500,nim250,nim350,nim500#maps and noise maps
                       ,w_250,w_350,w_500,p_src,fcat#header information from maps, prior_flux, output catalague structure
                       , bkg250,bkg350,bkg500,
                       sig_bkg250,sig_bkg350,sig_bkg500,outfile=None):#outfile saves pointing matrix and chains to a pickle file
#background estimates
    from scipy.sparse import coo_matrix
    import pystan
    import pickle

    # set up arrays to contain reconstructed map
    rmap250=np.empty((w_250._naxis1,w_250._naxis2))
    rmap350=np.empty((w_350._naxis1,w_350._naxis2))
    rmap500=np.empty((w_500._naxis1,w_500._naxis2))
    
    #there is code that removes sources which I have not included yet
    #
    x_pix,y_pix=np.meshgrid(np.arange(0,w_250._naxis1),np.arange(0,w_250._naxis2))
    
    #get pointing matrix
    amat_data,amat_row,amat_col,A,sx_pix,sy_pix,snoisy_map,ssig_map,snsrc,snpix=lstdrv_initsolveSP(sx250
                                                                                                   ,sy250,prf250,im250,nim250,x_pix,y_pix)
    #fit using stan
    fit_data,chains,iter=lstdrv_stan(amat_data,amat_row,amat_col,snoisy_map,ssig_map,snsrc,snpix,bkg250,sig_bkg250)
    
    
    #Get convergence stats
    R=np.empty((snsrc))
    for param in range(0,snsrc):
        fcat[1].data['R'][param]=convergence_stats(fit_data[:,:,param])[3]

    #Reshape fit to get median
    post=fit_data[:,:,0:snsrc].reshape((chains*iter/2.0,snsrc))
    #get Median flux values
    fcat[1].data['flux250']=np.array(quantileGet(50.0,post))
    fcat[1].data['el250']=np.array(quantileGet(30,post))#low lim
    fcat[1].data['eu250']=np.array(quantileGet(70,post))#up lim
    

    #Get convergence stats for background
    print convergence_stats(fit_data[:,:,snsrc])[3]
    post_bkg=fit_data[:,:,snsrc].reshape((chains*iter/2.0,1))
    #get Median and upper and lower limits for  bkg value
    bkg_250_med=np.array(quantileGet(50.0,post_bkg))
    bkg_250_low=np.array(quantileGet(30,post_bkg))#low lim
    bkg_250_up=np.array(quantileGet(70,post_bkg))#up lim
    print bkg_250_med, bkg_250_low, bkg_250_up


    #reconstruct map using median fluxes
    f_vec=np.empty((snsrc+1))
    f_vec[0:snsrc]=fcat[1].data['flux250']
    f_vec[-1]=bkg_250_med
    f=coo_matrix((f_vec, (range(0,snsrc+1),np.zeros(snsrc+1))), shape=(snsrc+1, 1))
    rmap_new=A*f

    
    #update reconstruction map
    rmap250[sx_pix,sy_pix]=np.asarray(rmap_new.todense()).reshape(-1)

    #save pointing matrix and chains if requested:
    if outfile != None:
        print("""Saving: \n
        1) pointing matrix (A) \n
        2) chains (chains) \n
        3) location of pixels in x (x_pix) \n
        4) location of pixels in y (ypix) \n 
        5) sigma (sig_pix) \n
        6) pixel data (ssig_map) \n
        7) number of sources being fitted (snsrc) \n
        8) number of pixels using for fit \n
        as a dictionary to %s""" % outfile)
        with open(outfile, 'wb') as f:
            pickle.dump({'A':A,'chains':fit_data,'x_pix':sx_pix,'y_pix':sy_pix,'sig_pix':snoisy_map,'im_pix':ssig_map,'snsrc':snsrc,'snpix':snpix}, f)
    return rmap250,fit_data,fcat


def cat_check_convert(inra,indec,wcs):
    """Checks sources in the prior list are within the boundaries of the map,
    and converts RA and DEC to pixel positions"""
    #get positions of sources in terms of pixels
    sx,sy=wcs.wcs_world2pix(inra,indec,0)
    #check if sources are within map 
    sgood=(sx > 0) & (sx < wcs._naxis1) & (sy > 0) & (sy < wcs._naxis2)# & np.isfinite(im250[np.rint(sx250).astype(int),np.rint(sy250).astype(int)])#this gives boolean array for cat
    #Redefine prior list so it only contains sources in the map
    sx=sx[sgood]
    sy=sy[sgood]
    sra=inra[sgood]
    sdec=indec[sgood]
    n_src=sgood.sum()
    return sx,sy,sra,sdec,n_src,sgood 
    
    



def create_XIDp_cat(nsrc,imhdu):
    """creates the XIDp catalogue in fits format required by HeDaM"""
    import datetime
    
    #----table info-----------------------
    #first define columns
    c1 = Column(name='XID', format='I', array=np.empty(nsrc,dtype=long))
    c2 = Column(name='ra', format='D', unit='degrees', array=np.empty(nsrc,dtype=float))
    c3 = Column(name='dec', format='D', unit='degrees', array=np.empty(nsrc,dtype=float))
    c4 = Column(name='flux250', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c5 = Column(name='flux250_err_u', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c6 = Column(name='flux250_err_l', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c7 = Column(name='flux350', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c8 = Column(name='flux350_err_u', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c9 = Column(name='flux350_err_l', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c10 = Column(name='flux500', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c11 = Column(name='flux500_err_u', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c12 = Column(name='flux500_err_l', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c13 = Column(name='bkg250', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c14 = Column(name='bkg350', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))
    c15 = Column(name='bkg500', format='E', unit='mJy', array=np.empty(nsrc,dtype=float))

    tbhdu = fits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15])
    
    tbhdu.header.set('TUCD1','XID',after='TUNIT1')      
    tbhdu.header.set('TDESC1','ID of source which corresponds to indexing of cov matrix.',after='TUCD1')         

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','250 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','250 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','250 Flux (at 25.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','350 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','350 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','350 Flux (at 25.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','500 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','500 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','500 Flux (at 25.9 percentile)',after='TUCD12')
    
    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['TITLE'] = 'XID catalogue'
    prihdr['Prior_C'] = prior_cat
    prihdr['TITLE']   = 'XID catalogue in '+ Field         
    prihdr['OBJECT']  = Field                               
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['VERSION'] = '1.0'                                 
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    #-----Covariance header---------------------------------
    c1 = Column(name='sigma_i_j_k', format='I', array=np.empty(nsrc,dtype=long))
    c2 = Column(name='XID_i', format='I', array=np.empty(nsrc,dtype=long))
    c3 = Column(name='XID_j', format='I', array=np.empty(nsrc,dtype=long))
    c4 = Column(name='XID_k', format='I', array=np.empty(nsrc,dtype=long))
    tbhdu = fits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15])


    thdulist = fits.HDUList([prihdu, tbhdu, covhdu, imhdu])
    return thdulist



