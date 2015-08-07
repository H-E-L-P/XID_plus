import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import XIDp_mod_beta as xid_mod
#----output folder-----------------
output_folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/'

hdulist=fits.open(output_folder+'catalogues/'+'master_catalogue.fits')
data=hdulist[1].data

import matplotlib
matplotlib.use('PDF')
from matplotlib import rc
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
pdf_pages=PdfPages("convergence_test.pdf")


bins=np.arange(0.8,2,0.05)
fig1,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,sharey=True)
n,bins,patches=ax1.hist(data['Rhat_250'],bins=bins,color='b')
ax1.axvline(x=1.2,ls='--')
ax1.set_yscale('log',nonposy='clip')
ind=bins[1:]>1.2
string='$\hat{R} > 1.2 =$'+str(int(n[ind].sum()))+r'/'+str(int(n.sum()))+' sources'
ax1.text(1.3,10E3,string)
ax1.set_ylim(1,60000)
n,bins,patches=ax2.hist(data['Rhat_350'],bins=bins,color='g')
ax2.axvline(x=1.2,ls='--')
ax2.set_yscale('log',nonposy='clip')
ind=bins[1:]>1.2
string='$\hat{R} > 1.2 =$'+str(int(n[ind].sum()))+r'/'+str(int(n.sum()))+' sources'
ax2.text(1.3,10E3,string)
ax2.set_ylim(1,60000)
n,bins,patches=ax3.hist(data['Rhat_500'],bins=bins,color='r')
ax3.axvline(x=1.2,ls='--')
ax3.set_yscale('log',nonposy='clip')
ind=bins[1:]>1.2
string='$\hat{R} > 1.2 =$'+str(int(n[ind].sum()))+r'/'+str(int(n.sum()))+' sources'
ax3.text(1.3,10E3,string)
ax3.set_ylim(1,60000)

ax1.set_ylabel(r'$\mathrm{N}_{250\mathrm{\mu m}}$')
ax2.set_ylabel(r'$\mathrm{N}_{350\mathrm{\mu m}}$')
ax3.set_ylabel(r'$\mathrm{N}_{500\mathrm{\mu m}}$')


ax3.set_xlabel(r'$\hat{R}$')
fig1.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig1.axes[:-1]],visible=False)
pdf_pages.savefig(fig1)

bins=np.arange(1,1200,10)
fig2,(ax1,ax2,ax3)=plt.subplots(3,sharex=True,sharey=True)
print n[ind].sum(),n.sum()
n,bins,patches=ax1.hist(data['n_eff_250'],bins=bins,color='b')
ax1.axvline(x=40,ls='--')
ind=bins[1:]<40
string='$n_{eff} < 40 =$'+str(int(n[ind].sum()))+r'/'+str(int(n.sum()))+' sources'
ax1.text(100,2E3,string)
ax2.set_yscale('log',nonposy='clip')
ax1.set_yscale('log',nonposy='clip')
ax1.set_ylim((1E0,5E4))
ax2.set_ylim((1E0,5E4))
ax3.set_ylim((1E0,5E4))

n,bins,patches=ax2.hist(data['n_eff_350'],bins=bins,color='g')
ax2.axvline(x=40,ls='--')
ind=bins[1:]<40
string='$n_{eff} < 40 =$'+str(int(n[ind].sum()))+r'/'+str(int(n.sum()))+' sources'
ax2.text(100,2E3,string)
n,bins,patches=ax3.hist(data['n_eff_500'],bins=bins,color='r')
ax3.axvline(x=40,ls='--')
ax3.set_yscale('log',nonposy='clip')
ind=bins[1:]<40
string='$n_{eff} < 40 =$'+str(int(n[ind].sum()))+r'/'+str(int(n.sum()))+' sources'
ax3.text(100,2E3,string)
ax3.set_xlabel(r'$n_{eff}$')
fig2.subplots_adjust(hspace=0)
ax1.set_ylabel(r'$\mathrm{N}_{250\mathrm{\mu m}}$')
ax2.set_ylabel(r'$\mathrm{N}_{350\mathrm{\mu m}}$')
ax3.set_ylabel(r'$\mathrm{N}_{500\mathrm{\mu m}}$')
plt.setp([a.get_xticklabels() for a in fig2.axes[:-1]],visible=False)
pdf_pages.savefig(fig2)

pdf_pages.close()
