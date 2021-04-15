__author__ = 'pdh21'
from astropy.io import fits
import scipy.stats as st
import numpy as np
from astropy.io import fits


def ymod_map(prior,flux):
    """Create replicated model map (no noise or background) i.e. A*f

    :param prior: xidplus.prior class
    :param flux: flux vector
    :return: map array, in same format as prior.sim
    """
    from scipy.sparse import coo_matrix

    f=coo_matrix((flux, (range(0,prior.nsrc),np.zeros(prior.nsrc))), shape=(prior.nsrc, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc))
    rmap_temp=(A*f)
    return np.asarray(rmap_temp.todense())


def Bayesian_pvals(prior,post_rep_map):
    """Get Bayesian P values for each pixel

    :param prior: xidplus.prior class
    :param post_rep_map: posterior replicated maps
    :return: Bayesian P values
    """
    pval=np.empty_like(prior.sim)
    for i in range(0,prior.snpix):
        ind=post_rep_map[i,:]<prior.sim[i]
        pval[i]=sum(ind)/np.float(post_rep_map.shape[1])
    pval[np.isposinf(pval)]=1.0
    #pval[np.isneginf(pval)]=0.0
    return pval

def make_Bayesian_pval_maps(prior,post_rep_map):
    """Bayesian P values, quoted as sigma level

    :param prior: xidplus.prior class
    :param post_rep_map: posterior replicated maps
    :return: Bayesian P values converted to sigma level
    """
    pval=Bayesian_pvals(prior,post_rep_map)
    for i in range(0,prior.snpix):
        pval[i]=st.norm.ppf(pval[i])
    pval[np.isposinf(pval)]=6.0
    pval[np.isneginf(pval)]=-6.0
    return pval


def Bayes_Pval_res(prior,post_rep_map):
    """The local Bayesian P value residual statistic. 
    
    
    :param prior: xidplus.prior class
    :param post_rep_map: posterior replicated maps
    :return: Bayesian P value residual statistic for each source
    """
    Bayes_pval_res_vals=np.empty((prior.nsrc))
    for i in range(0,prior.nsrc):
        ind= prior.amat_col == i
        t = np.sum(((post_rep_map[prior.amat_row[ind], :] - prior.sim[prior.amat_row[ind], None]) / (
        np.sqrt(2) * prior.snim[prior.amat_row[ind], None])) ** 2.0, axis=0)
        ind_T = t / ind.sum() > 2
        Bayes_pval_res_vals[i] = ind_T.sum()/np.float(post_rep_map.shape[1])

    return Bayes_pval_res_vals



def make_fits_image(prior,pixel_values):
    """Make FITS image realting to map in xidplus.prior class
    :param prior: xidplus.prior class
    :param pixel_values: pixel values in format of xidplus.prior.sim
    :return: FITS hdulist
    """
    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    data=np.full((y_range,x_range),np.nan)
    data[prior.sy_pix-np.min(prior.sy_pix)-1,prior.sx_pix-np.min(prior.sx_pix)-1]=pixel_values
    hdulist = fits.HDUList([fits.PrimaryHDU(header=prior.imphdu),fits.ImageHDU(data=data,header=prior.imhdu)])
    hdulist[1].header['CRPIX1']=hdulist[1].header['CRPIX1']-np.min(prior.sx_pix)-1
    hdulist[1].header['CRPIX2']=hdulist[1].header['CRPIX2']-np.min(prior.sy_pix)-1

    return hdulist


def replicated_maps(priors,posterior,nrep=1000):
    """Create posterior replicated maps

    :param priors: list of xidplus.prior class
    :param posterior: xidplus.posterior class
    :param nrep: number of replicated maps
    :return: 
    """

    #check nrep is less than number of samples
    if nrep>posterior.samples['bkg'].shape[0]:
        nrep=posterior.samples['bkg'].shape[0]
    mod_map_array=list(map(lambda prior:np.empty((prior.snpix,nrep)), priors))
    for i in range(0,nrep):
        try:
            for b in range(0,len(priors)):
                mod_map_array[b][:,i]= ymod_map(priors[b],posterior.samples['src_f'][i,b,:]).reshape(-1)\
                                       +posterior.samples['bkg'][i,b]\
                                       +np.random.normal(scale=np.sqrt(priors[b].snim**2
                                                                       +posterior.samples['sigma_conf'][i,b]**2))
        except IndexError:
            for b in range(0,len(priors)):
                mod_map_array[b][:,i]= ymod_map(priors[b],posterior.samples['src_f'][i,b,:]).reshape(-1)\
                                       +posterior.samples['bkg'][i]\
                                       +np.random.normal(scale=np.sqrt(priors[b].snim**2
                                                                       +posterior.samples['sigma_conf'][i]**2))
    return mod_map_array