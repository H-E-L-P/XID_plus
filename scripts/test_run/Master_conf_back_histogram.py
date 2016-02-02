import pickle
import dill
import numpy as np
from xidplus import moc_routines
import xidplus
import matplotlib
matplotlib.use('PDF')
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
output_folder='./'

pdf_pages=PdfPages("sig_back.pdf")
with open(output_folder+'Master_prior.pkl', "rb") as f:
    Master = pickle.load(f)

tiles=Master['tiles']
order=Master['order']
prior250=Master['psw']
sig_back=np.empty((3000.0*len(tiles),6))
250_sig_im=prior250.im
350_sig_im=prior350.im
500_sig_im=prior500.im
250_back_im=prior250.im
350_back_im=prior350.im
500_back_im=prior500.im


for i in range(0,len(tiles)):
    print 'On tile '+str(i)+' out of '+str(len(tiles))
    infile=output_folder+'Lacy_test_file_'+str(tiles[i])+'_'+str(order)+'.pkl'
    with open(infile, "rb") as f:
        obj = pickle.load(f)
    tmp_prior250=obj['psw']
    tmp_prior350=obj['pmw']
    tmp_prior500=obj['plw']
    tmp_posterior=obj['posterior']
    sig_back_ind=np.array([tmp_prior250.nsrc,2*tmp_prior250.nsrc+1,3*tmp_prior250.nsrc+2,3*tmp_prior250.nsrc+3,3*tmp_prior250.nsrc+4,3*tmp_prior250.nsrc+5])
    sig_back[i*3000.0:(i*3000.0)+3000]=tmp_posterior.stan_fit[:,:,sig_back_ind].reshape(750*4,6)


    ra,dec=wcs_temp.wcs_pix2world(tmp_prior250.sx_pix,tmp_prior250.sy_pix,0)
    ind_map=np.array(moc_routines.check_in_moc(ra,dec,,keep_inside=True))
    kept_sources=moc_routines.sources_in_tile(tiles[i],order,ra,dec)


fig1,ax1=plt.subplots(figsize=(10,10))
ax1.hist(sig_back[:,0],bins=np.arange(-8,4,0.1),color='b',alpha=0.5,label=r'$250 \mathrm{\mu m}$')
ax1.hist(sig_back[:,1],bins=np.arange(-8,4,0.1),color='g',alpha=0.5,label=r'$350 \mathrm{\mu m}$')
ax1.hist(sig_back[:,2],bins=np.arange(-8,4,0.1),color='r',alpha=0.5,label=r'$500 \mathrm{\mu m}$')
plt.tight_layout()
plt.legend(loc='lower left')
ax1.set_xlabel(r'Background $(\mathrm{mJy})$')
pdf_pages.savefig(fig1)
fig2,ax2=plt.subplots(figsize=(10,10))
ax2.hist(sig_back[:,3],bins=np.arange(1,4,0.01),color='b',alpha=0.5,label=r'$250 \mathrm{\mu m}$')
ax2.hist(sig_back[:,4],bins=np.arange(1,4,0.01),color='g',alpha=0.5,label=r'$350 \mathrm{\mu m}$')
ax2.hist(sig_back[:,5],bins=np.arange(1,4,0.01),color='r',alpha=0.5,label=r'$500 \mathrm{\mu m}$')
plt.tight_layout()
plt.legend(loc='lower left')
ax2.set_xlabel(r'$\sigma_{confusion}(\mathrm{mJy})$')
pdf_pages.savefig(fig2)
pdf_pages.close()

    
