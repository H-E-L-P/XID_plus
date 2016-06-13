import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs

import pickle
import dill
import xidplus
import matplotlib
matplotlib.use('PDF')
import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages

#folder='/research/astro/fir/HELP/XID_plus_output/100micron/log_prior_flux/'
folder='/Users/pdh21/HELP/XID_plus_output/100micron/conf_noise/uniform_prior/'
#'/research/astro/fir/HELP/XID_plus_output/100micron/log_uniform_prior_test/old/'
infile=folder+'Master_prior.pkl'
with open(infile, "rb") as f:
    obj = pickle.load(f)
prior250=obj['psw']
prior350=obj['pmw']    
prior500=obj['plw']

folder='/Users/pdh21/HELP/XID_plus_output/100micron/conf_noise/uniform_prior/'
infile=folder+'master_posterior.pkl'
with open(infile, "rb") as f:
    obj = pickle.load(f)
posterior=obj['posterior']

pdf_pages=PdfPages("colour_colour_sim_all.pdf")


nsrc=prior250.nsrc

(samples,chains,params)=posterior.shape
flattened_post=posterior.reshape(samples*chains,params)
pctiles=np.percentile(flattened_post, [16,50,84], axis=0)
flattened_post=np.log10(flattened_post)
med_250=pctiles[1,0:prior250.nsrc]
med_350=pctiles[1,prior250.nsrc+1:2*prior250.nsrc+1]
med_500=pctiles[1,2*prior250.nsrc+2:3*prior250.nsrc+2]
id=(med_250>4) & (med_350>4) & (med_500>6)

fig,ax=plt.subplots(figsize=(5.5,5))
c350_250=np.power(10.0,flattened_post[:,nsrc+1:(2*nsrc)+1][:,:].reshape(-1))/np.power(10.0,flattened_post[:,0:nsrc][:,:].reshape(-1))
c500_350=np.power(10.0,flattened_post[:,2*nsrc+2:(3*nsrc)+2][:,:].reshape(-1))/np.power(10.0,flattened_post[:,nsrc+1:(2*nsrc)+1][:,:].reshape(-1))



tmp=ax.hexbin(c350_250,c500_350, gridsize=200, extent=(-0.8,0.7,-0.8,0.7),yscale='log',xscale='log',bins='log',cmap=plt.get_cmap('Greys'))
ax.set_xlabel(r'$S_{350 \mathrm{\mu m}}/S_{250 \mathrm{\mu m}}$')
ax.set_ylabel(r'$S_{500 \mathrm{\mu m}}/S_{350 \mathrm{\mu m}}$')




#SED track---------------------------------------------

from astropy.io import ascii
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import interp1d




# In[2]:

M82 = ascii.read("/Users/pdh21/astrodata/SEDs/Berta2013/templates_berta_norm_LIR/Blue_SF_glx.norm_LIR")#ascii.read("./M82_template_norm.sed")
arp220 = ascii.read("/Users/pdh21/astrodata/SEDs/Berta2013/templates_berta_norm_LIR/Red_SF_glx_1.norm_LIR")#ascii.read("./Arp220_template_norm.sed")


# In[3]:

class FilterDefinition:
    def __init__(self):
        """
        Placeholder for the filter definition information.
        """
        self.name = None
        self.wavelength = None
        self.transmision = None

class FilterFile:
    def __init__(self, file='FILTER.RES.v8.R300'):
        """
        Read a EAZY (HYPERZ) filter file.
        """
        fp = open(file)
        lines = fp.readlines()
        fp.close()
        
        filters = []
        wave = []
        for line in lines:
            try:
                lspl = np.cast[float](line.split())
                wave.append(lspl[1])
                trans.append(lspl[2])
            except (ValueError,IndexError):
                if len(wave) > 0:
                    new_filter = FilterDefinition()
                    new_filter.name = header
                    new_filter.wavelength = np.cast[float](wave)
                    new_filter.transmission = np.cast[float](trans)
                    filters.append(new_filter)
                    
                header = ' '.join(line.split()[1:])
                wave = []
                trans = []
        # last one
        new_filter = FilterDefinition()
        new_filter.name = header
        new_filter.wavelength = np.cast[float](wave)
        new_filter.transmission = np.cast[float](trans)
        filters.append(new_filter)
           
        self.filters = filters
        self.NFILT = len(filters)
    
    def names(self):
        """
        Print the filter names.
        """
        for i in range(len(self.filters)):
            print '%5d %s' %(i+1, self.filters[i].name)
    
    def write(self, file='xxx.res', verbose=True):
        """
        Dump the filter information to a filter file.
        """
        fp = open(file,'w')
        for filter in self.filters:
            fp.write('%6d %s\n' %(len(filter.wavelength), filter.name))
            for i in range(len(filter.wavelength)):
                fp.write('%-6d %.5e %.5e\n' %(i+1, filter.wavelength[i], filter.transmission[i]))
        
        fp.close()
        if verbose:
            print 'Wrote <%s>.' %(file)
            
    def search(self, search_string, case=True, verbose=True):
        """ 
        Search filter names for `search_string`.  If `case` is True, then
        match case.
        """
        import re
        
        if not case:
            search_string = search_string.upper()
        
        matched = []
        
        for i in range(len(self.filters)):
            filt_name = self.filters[i].name
            if not case:
                filt_name = filt_name.upper()
                
            if re.search(search_string, filt_name) is not None:
                if verbose:
                    print '%5d %s' %(i+1, self.filters[i].name)
                matched.append(i)
        
        return np.array(matched)


# In[4]:

filter=FilterFile(file="/Users/pdh21/astrodata/COSMOS/WP5_COSMOS_XIDplus_P2/filters.res")


# In[5]:

SPIRE_250=filter.filters[215]
SPIRE_350=filter.filters[216]
SPIRE_500=filter.filters[217]


# I have used the [KINGFISH](http://www.astro.princeton.edu/~draine/Notes_re_KINGFISH_SPIRE_Photometry.pdf) document as a basis of what to do fo integrating model with flux

# In[20]:

def fnu_filt(sed_fnu,filt_nu,filt_trans,nu_0,sed_f0):
    #f_nu=Int(d_nu f_nu R_nu)/Int(d_nu (nu/nu_0)^-1 R_nu)
    numerator=np.trapz(sed_fnu*filt_trans,x=filt_nu)
    denominator=np.trapz(filt_trans*(nu_0/filt_nu),x=filt_nu)
    
    #colour correction
    #C=Int(d_nu (nu/nu_0)^-1 R_nu)/Int(d_nu (f(nu)/f(nu_0)) R_nu)
    C_num=np.trapz(filt_trans*(nu_0/filt_nu),x=filt_nu)
    C_denom=np.trapz(filt_trans*(sed_fnu/sed_f0),x=filt_nu)

    
    return (numerator/denominator)#*(C_num/C_denom)


# In[25]:

flux=arp220['col2']*(arp220['col1']**2)/3E8
wave=arp220['col1']/1E4
S_250=[]
S_350=[]
S_500=[]
red=[]
for z in np.arange(1,8.5,0.1):
    sed=interp1d((z+1.0)*wave, flux)
    S_250.append(fnu_filt(sed(SPIRE_250.wavelength/1E4),3E8/(SPIRE_250.wavelength/1E10),SPIRE_250.transmission,3E8/250E-6,sed(250.0)))
    S_350.append(fnu_filt(sed(SPIRE_350.wavelength/1E4),3E8/(SPIRE_350.wavelength/1E10),SPIRE_350.transmission,3E8/350E-6,sed(350.0)))
    S_500.append(fnu_filt(sed(SPIRE_500.wavelength/1E4),3E8/(SPIRE_500.wavelength/1E10),SPIRE_500.transmission,3E8/500E-6,sed(500.0)))
    red.append(z)
    
points = np.array([np.array(S_350)/np.array(S_250),np.array(S_500)/np.array(S_350)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=plt.get_cmap('rainbow'),norm=plt.Normalize(0, 9),linestyles = 'solid')
lc.set_array(np.array(red))
lc.set_linewidth(5)
plt.gca().add_collection(lc)

flux=M82['col2']*((M82['col1']/1E10)**2)/3E8
wave=M82['col1']/1E4
S_250=[]
S_350=[]
S_500=[]
red=[]
for z in np.arange(1,8.5,0.1):
    sed=interp1d((z+1.0)*wave, flux)
    S_250.append(fnu_filt(sed(SPIRE_250.wavelength/1E4),3E8/(SPIRE_250.wavelength/1E10),SPIRE_250.transmission,3E8/250E-6,sed(250.0)))
    S_350.append(fnu_filt(sed(SPIRE_350.wavelength/1E4),3E8/(SPIRE_350.wavelength/1E10),SPIRE_350.transmission,3E8/350E-6,sed(350.0)))
    S_500.append(fnu_filt(sed(SPIRE_500.wavelength/1E4),3E8/(SPIRE_500.wavelength/1E10),SPIRE_500.transmission,3E8/500E-6,sed(500.0)))
    red.append(z)
    
points = np.array([np.array(S_350)/np.array(S_250),np.array(S_500)/np.array(S_350)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap=plt.get_cmap('rainbow'),norm=plt.Normalize(0, 9),linestyles = 'dashed')
lc.set_array(np.array(red))
lc.set_linewidth(3)
plt.gca().add_collection(lc)

plt.xlim(0,5)
plt.ylim(0,5)

axcb=plt.colorbar(lc)
axcb.set_label('Redshift')


#---------------------------------------------------------







#fig.colorbar(tmp, ax=ax)
pdf_pages.savefig(fig)
#fig2,ax2=plt.subplots(figsize=(5.5,5))
#tmp=ax2.hexbin(np.power(10.0,flattened_post[:,0:nsrc].reshape(-1)),c500_350, gridsize=40, extent=(1,30.0,1.0E-4,3),yscale='log',cmap=plt.get_cmap('Reds'))
#ax2.set_xlabel(r'$S_{250 \mathrm{\mu m}}} (\mathrm{mJy}))$')
#ax2.set_ylabel(r'$S_{500 \mathrm{\mu m}}/S_{350 \mathrm{\mu m}}$')
#fig2.colorbar(tmp, ax=ax2)
#pdf_pages.savefig(fig2)

pdf_pages.close()
