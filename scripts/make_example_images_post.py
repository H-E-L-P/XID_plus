import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import sys

sys.path.append('/research/astro/fir/HELP/XID_plus/')
import XIDp_mod_beta as xid_mod
import os
import matplotlib
matplotlib.use('pdf')
import pylab as plt
#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'

with open(output_folder+'Tiling_info.pkl', "rb") as f:
        obj = pickle.load(f)
tiles=obj['tiles']
tiling_list=obj['tiling_list']
nsources=tiling_list.shape[0]
sources_percentile=np.empty((nsources,14))

infile=output_folder+'Lacey_log10_norm1_149_5p1_6.pkl'
with open(infile, "rb") as f:
    dictname = pickle.load(f)
prior250=dictname['psw']
prior350=dictname['pmw']    
prior500=dictname['plw']
print '----wcs----'
print prior250.wcs._naxis1,prior250.wcs._naxis2
print '---------------'
posterior=dictname['posterior']
posterior.stan_fit=np.power(10.0,posterior.stan_fit)
posterior.stan_fit[:,:,[prior250.nsrc,2*prior250.nsrc+1,3*prior250.nsrc+1]]=np.log10(posterior.stan_fit[:,:,[prior250.nsrc,2*prior250.nsrc+1,3*prior250.nsrc+2]])



#flatten chains------
samples,chains,params=posterior.stan_fit.shape
flattened_post=posterior.stan_fit.reshape(samples*chains,params)



#Folder containing maps
imfolder='/research/astro/fir/cclarke/lacey/released/'
#field
field='COSMOS'
#SMAP version
SMAPv='4.2'


# In[7]:

pswfits=imfolder+'cosmos_itermap_lacey_07012015_simulated_observation_w_noise_PSW_hipe.fits.gz'#SPIRE 250 map
#-----250-------------
hdulist = fits.open(pswfits)
im250phdu=hdulist[0].header
fits_template=hdulist[1]
im250=hdulist[1].data*1.0E3
nim250=hdulist[2].data*1.0E3
w_250 = wcs.WCS(hdulist[1].header)
pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
hdulist.close()

#match to sources I am interested in

c= SkyCoord(ra=prior250.sra*u.degree,dec=prior250.sdec*u.degree)
c1=SkyCoord(ra=np.array([149.5976,149.5971])*u.degree,dec=np.array([1.716100,1.716099])*u.degree)
idx,d2d,d3d,= c1.match_to_catalog_sky(c)
xid=np.random.multivariate_normal(np.array([0.0,49.0404]),np.array([[15.634,0],[0,1.444]]),5000)


def yrep_map(prior,fvec):
    from scipy.sparse import coo_matrix
    

    x_range=np.max(prior.sx_pix)-np.min(prior.sx_pix)
    y_range=np.max(prior.sy_pix)-np.min(prior.sy_pix)
    f=coo_matrix((fvec, (range(0,prior.nsrc+1),np.zeros(prior.nsrc+1))), shape=(prior.nsrc+1, 1))
    A=coo_matrix((prior.amat_data, (prior.amat_row, prior.amat_col)), shape=(prior.snpix, prior.nsrc+1))
    rmap_temp=(A*f)
    pred_map=np.empty_like(prior.im)
    pred_map[:,:]=0.0#prior.im
    pred_map_noise=np.empty_like(prior.im)
    pred_map_noise[:,:]=0.0#prior.im
    pred_map[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)#+np.random.randn(prior.snpix)*prior.snim
    pred_map_noise[prior.sy_pix,prior.sx_pix]=np.asarray(rmap_temp.todense()).reshape(-1)+np.random.randn(prior.snpix)*prior.snim

    return pred_map,pred_map_noise
samples,chains,params=posterior.stan_fit.shape
flattened_post=posterior.stan_fit.reshape(samples*chains,params)

#import matplotlib
#matplotlib.use('PS')
#import pylab as plt
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/maps/'
import triangle
for i in range(0,1):#samples*chains):
    print 'making map '+ str(i) 
    figure,ax=plt.subplots(nrows=1,ncols=5)
    pred_map,pred_map_noise=yrep_map(prior250,flattened_post[i,0:prior250.nsrc+1])
    ax[0].imshow(pred_map,interpolation='nearest',vmin=-10,vmax=50)
    ax[0].plot(prior250.sx[idx[0]], prior250.sy[idx[0]],'s',mfc='none',ms=10,mec='k')
    ax[0].set_ylim(np.min(prior250.sy_pix),np.max(prior250.sy_pix))
    ax[0].set_xlim(np.min(prior250.sx_pix),np.max(prior250.sx_pix))
    ax[0].set_xticklabels(())
    ax[0].set_yticklabels(())
    ax[1].imshow(pred_map_noise,interpolation='nearest',vmin=-10,vmax=50)
    ax[1].set_ylim(np.min(prior250.sy_pix),np.max(prior250.sy_pix))
    ax[1].set_xlim(np.min(prior250.sx_pix),np.max(prior250.sx_pix))
    ax[1].set_xticklabels(())
    ax[1].set_yticklabels(())
    ax[2].imshow((im250),interpolation='nearest',vmin=-10,vmax=50)
    ax[2].set_ylim(np.min(prior250.sy_pix),np.max(prior250.sy_pix))
    ax[2].set_xlim(np.min(prior250.sx_pix),np.max(prior250.sx_pix))
    ax[2].set_xticklabels(())
    ax[2].set_yticklabels(())
    im=ax[3].imshow((im250)-(pred_map),interpolation='nearest',vmin=-10,vmax=50)
    ax[3].set_ylim(np.min(prior250.sy_pix),np.max(prior250.sy_pix))
    ax[3].set_xlim(np.min(prior250.sx_pix),np.max(prior250.sx_pix))
    ax[3].set_xticklabels(())
    ax[3].set_yticklabels(()) 
    ax[4].imshow((pred_map),interpolation='nearest',vmin=-10,vmax=50)
    ax[4].plot(prior250.sx[idx], prior250.sy[idx],'o',mfc='none', mec='k', ms=10)
    ax[4].set_ylim(prior250.sy[idx[0]]-10,prior250.sy[idx[1]]+10)
    ax[4].set_xlim(prior250.sx[idx[0]]-10,prior250.sx[idx[1]]+10)
    ax[4].set_xticklabels(())
    ax[4].set_yticklabels(())

    fig = triangle.corner(flattened_post[:,idx],extents=[(0,60),(0,60)], color='g',labels=[r"Source $1$ $S_{250\mathrm{\mu m}}$", r"Source $2$ $S_{250\mathrm{\mu m}}$"])
    #fig = triangle.corner(flattened_post[:,idx],truths=flattened_post[i,idx],extents=[(0,60),(0,60)], color='g',labels=[r"Source $1$ $S_{250\mathrm{\mu m}}$", r"Source $2$ $S_{250\mathrm{\mu m}}$"]) ## if you want particuakr sample to be plotted as truth
    fig = triangle.corner(xid,truths=np.array([33.44,22.51]),extents=[(0,60),(0,60)],fig=fig,labels=[r"Source $1$ $S_{250\mathrm{\mu m}}$ (mJy)", r"Source $2$ $S_{250\mathrm{\mu m}}$ (mJy)"], color='b')
    figure.subplots_adjust(bottom=0.05)
    cbar_ax = figure.add_axes([0.1,0.15,0.8,0.07])
    cbar_ax.set_title('mJy')
    figure.colorbar(im,cax=cbar_ax, orientation='horizontal')

    #figure.savefig(output_folder+'example_image_'+str(i)+'.pdf')
    figure.clf()
    fig.savefig('/research/astro/fir/HELP/XID_plus/Paper/example_tri_DESHPOT_XIDp.pdf')
    fig.clf()
