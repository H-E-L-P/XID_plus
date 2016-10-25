__author__ = 'pdh21'
from astropy.io import fits
import scipy.stats as st
import numpy as np
from astropy.io import fits

def ymod_map(prior,posterior_sample):
    from scipy.sparse import coo_matrix

    f=coo_matrix((posterior_sample[0:prior.nsrc], (range(0,prior.nsrc),np.zeros(prior.nsrc))), shape=(prior.nsrc, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc))
    rmap_temp=(A*f)
    #pred_map=np.empty_like(prior.im)
    #pred_map[:,:]=0.0
    #pred_map[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)#+np.random.randn(prior.snpix)*prior.snim

    return np.asarray(rmap_temp.todense())


def post_rep_map(prior,mod_map,back,conf_noise):
    return mod_map+back+np.random.normal(scale=np.sqrt(prior.snim**2+conf_noise**2))

def Bayesian_pvals(prior,post_rep_map):
    pval=np.empty_like(prior.sim)
    for i in range(0,prior.snpix):
        ind=post_rep_map[i,:]<prior.sim[i]
        pval[i]=sum(ind)/np.float(post_rep_map.shape[1])
    pval[np.isposinf(pval)]=1.0
    #pval[np.isneginf(pval)]=0.0
    return pval

def make_Bayesian_pval_maps(prior,post_rep_map):
    pval=Bayesian_pvals(prior,post_rep_map)
    for i in range(0,prior.snpix):
        pval[i]=st.norm.ppf(pval[i])
    pval[np.isposinf(pval)]=6.0
    pval[np.isneginf(pval)]=np.nan
    return pval

def moments_of_pval_dist(pval):
    moments=st.moment(pval,moment=np.array([1,2,3,4]))
    moments[0]=np.mean(pval)
    return moments

def Bayes_Pval_res(prior,post_rep_map):
    Bayes_pval_res_vals=np.empty((prior.nsrc))
    for i in range(0,prior.nsrc):
        ind= prior.amat_col == i

        T_data=np.sum((((prior.sim[prior.amat_row[ind]]-np.median(post_rep_map[prior.amat_row[ind],:],axis=1))/prior.snim[prior.amat_row[ind]])**2))
        T_rep=np.sum((((post_rep_map[prior.amat_row[ind],:]-np.median(post_rep_map[prior.amat_row[ind],:],axis=1)[:,None])/np.std(post_rep_map[prior.amat_row[ind],:],axis=1)[:,None])**2),axis=0)
        ind_T=T_data > 2*T_rep
        Bayes_pval_res_vals[i]=sum(ind_T)/np.float(post_rep_map.shape[1])
    return Bayes_pval_res_vals

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
