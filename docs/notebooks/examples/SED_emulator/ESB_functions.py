from astropy.io import ascii, fits
from astropy.table import QTable, Table
import arviz as az
from astropy.coordinates import SkyCoord
from astropy import units as u
import os

import pymoc
from astropy import wcs
from astropy.table import vstack, hstack

import numpy as np
import xidplus


# # Applying XID+CIGALE to Extreme Starbursts
# In this notebook, we read in the data files and prepare them for fitting with XID+CIGALE, the SED prior model extension to XID+. Here we focus on sources in [Rowan-Robinson et al. 2018](https://arxiv.org/abs/1704.07783) and claimed to have a star formation rate of $> 10^{3}\mathrm{M_{\odot}yr^{-1}}$

# In[2]:
def process_prior(c,new_Table=None,
                  path_to_data=['../../../data/'],
                  field=['Lockman-SWIRE'],
                  path_to_SPIRE=['/Volumes/pdh_storage/dmu_products/dmu19/dmu19_HELP-SPIRE-maps/data/'],
                  redshift_file=["/Volumes/pdh_storage/dmu_products/dmu24/dmu24_Lockman-SWIRE/data/master_catalogue_Lockman-SWIRE_20170710_photoz_20170802_r_and_irac1_optimised_UPDATED_IDs_20180219.fits"],
                  redshift_prior=[0.1,2.0],
                  radius=6.0,
                  alt_model=False):



    # Import required modules

    # In[3]:



    # In[4]:



    # Set image and catalogue filenames

    # In[5]:

    #Folder containing maps
    pswfits=path_to_SPIRE[0]+'{}_SPIRE250_v1.0.fits'.format(field[0])#SPIRE 250 map
    pmwfits=path_to_SPIRE[0]+'{}_SPIRE350_v1.0.fits'.format(field[0])#SPIRE 350 map
    plwfits=path_to_SPIRE[0]+'{}_SPIRE500_v1.0.fits'.format(field[0])#SPIRE 500 map




    #output folder
    output_folder='./'


    # Load in images, noise maps, header info and WCS information

    # In[6]:

    #-----250-------------
    hdulist = fits.open(pswfits)
    im250phdu=hdulist[0].header
    im250hdu=hdulist[1].header

    im250=hdulist[1].data*1.0E3 #convert to mJy
    nim250=hdulist[3].data*1.0E3 #convert to mJy
    w_250 = wcs.WCS(hdulist[1].header)
    pixsize250=np.abs(3600.0*w_250.wcs.cdelt[0]) #pixel size (in arcseconds)
    hdulist.close()
    #-----350-------------
    hdulist = fits.open(pmwfits)
    im350phdu=hdulist[0].header
    im350hdu=hdulist[1].header

    im350=hdulist[1].data*1.0E3 #convert to mJy
    nim350=hdulist[3].data*1.0E3 #convert to mJy
    w_350 = wcs.WCS(hdulist[1].header)
    pixsize350=np.abs(3600.0*w_350.wcs.cdelt[0]) #pixel size (in arcseconds)
    hdulist.close()
    #-----500-------------
    hdulist = fits.open(plwfits)
    im500phdu=hdulist[0].header
    im500hdu=hdulist[1].header
    im500=hdulist[1].data*1.0E3 #convert to mJy
    nim500=hdulist[3].data*1.0E3 #convert to mJy
    w_500 = wcs.WCS(hdulist[1].header)
    pixsize500=np.abs(3600.0*w_500.wcs.cdelt[0]) #pixel size (in arcseconds)
    hdulist.close()


    # XID+ uses Multi Order Coverage (MOC) maps for cutting down maps and catalogues so they cover the same area. It can also take in MOCs as selection functions to carry out additional cuts. Lets use the python module [pymoc](http://pymoc.readthedocs.io/en/latest/) to create a MOC, centered on a specific position we are interested in. We will use a HEALPix order of 15 (the resolution: higher order means higher resolution)




    moc=pymoc.util.catalog.catalog_to_moc(c,100,15)


    # Load in catalogue you want to fit (and make any cuts). Here we use HELP's VO database and directly call it using PyVO

    # In[10]:

    import pyvo as vo
    service = vo.dal.TAPService("https://herschel-vos.phys.sussex.ac.uk/__system__/tap/run/tap")


    # In[11]:

    resultset = service.search("SELECT TOP 10000 * FROM herschelhelp.main WHERE 1=CONTAINS(POINT('ICRS', ra, dec),CIRCLE('ICRS',"+str(c.ra.deg[0])+", "+str(c.dec.deg[0])+", 0.028 ))")


    # In[12]:

    masterlist=resultset.table


    def construct_prior(Table=None):
        from astropy.coordinates import SkyCoord
        #first use standard cut (i.e. not star and is detected in at least 3 opt/nir bands)
        prior_list=masterlist[(masterlist['flag_gaia']!=3) & (masterlist['flag_optnir_det']>=3)]

        #make skycoord from masterlist
        catalog=SkyCoord(ra=masterlist['ra'],dec=masterlist['dec'])
        #make skycoord from input table
        c = SkyCoord(ra=Table['ra'], dec=Table['dec'])
        #search around all of the new sources
        idxc, idxcatalog, d2d, d3d=catalog.search_around_sky(c,radius*u.arcsec)

        #for every new sources
        for src in range(0,len(Table)):
            #limit to matches around interested sources
            ind = idxc == src
            #if there are matches
            if ind.sum() >0:
                #choose the closest and check if its in the prior list all ready
                in_prior=prior_list['help_id']==masterlist[idxcatalog][ind][np.argmin(d2d[ind])]['help_id']

                #if its not in prior list
                if in_prior.sum() <1:
                    print(in_prior.sum())
                    #add to appended sources
                    prior_list=vstack([prior_list,masterlist[idxcatalog][ind][np.argmin(d2d[ind])]])



        return prior_list


    # In[64]:

    import astropy.units as u
    #create table of candidate source
    t = QTable([c.ra, c.dec], names=('ra', 'dec'))
    #add candidate source to new sources table, create prior list
    if new_Table is not None:
        prior_list=construct_prior(vstack([t,new_Table]))
    else:
        prior_list = construct_prior(t)

    if alt_model==True:
        sep = 18
        separation = c.separation(SkyCoord(prior_list['ra'], prior_list['dec'])).arcsec
        remove_ind = (separation > np.min(separation)) & (separation < sep)
        prior_list.remove_rows(remove_ind)

    # ## Get Redshift and Uncertianty
    #

    # Ken Duncan defines a median and a hierarchical bayes combination redshift. We need uncertianty so lets match via `help_id`

    # In[26]:

    photoz=Table.read(redshift_file[0])


    # In[27]:

    #help_id=np.empty((len(photoz)),dtype=np.dtype('U27'))
    for i in range(0,len(photoz)):
        photoz['help_id'][i]=str(photoz['help_id'][i].strip()).encode('utf-8')
    #photoz['help_id']=help_id


    # In[28]:

    from astropy.table import Column, MaskedColumn
    prior_list['redshift']=MaskedColumn(np.full((len(prior_list)),fill_value=redshift_prior[0]),mask=[False]*len(prior_list))
    prior_list.add_column(MaskedColumn(np.full((len(prior_list)),fill_value=redshift_prior[1]),mask=[False]*len(prior_list),name='redshift_unc'))


    # In[29]:

    photoz


    # In[30]:

    ii=0
    for i in range(0,len(prior_list)):
        ind=photoz['help_id'] == prior_list['help_id'][i]
        try:
            if photoz['z1_median'][ind]>0.0:
                prior_list['redshift'][i]=photoz['z1_median'][ind]
                prior_list['redshift_unc'][i]=np.max(np.array([np.abs(photoz['z1_median'][ind]-photoz['z1_min'][ind]),np.abs(photoz['z1_max'][ind]-photoz['z1_median'][ind])]))

            #prior_list['redshift_unc'].mask[i]=False
            #prior_list['redshift'].mask[i]=False

        except ValueError:
            None


    # In[33]:

    dist_matrix=np.zeros((len(prior_list),len(prior_list)))
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    for i in range(0,len(prior_list)):
        for j in range(0,len(prior_list)):
            if i>j:
                coord1 = SkyCoord(ra=prior_list['ra'][i]*u.deg,dec=prior_list['dec'][i]*u.deg,frame='icrs')

                coord2=SkyCoord(ra=prior_list['ra'][j]*u.deg,dec=prior_list['dec'][j]*u.deg)
                dist_matrix[i,j] = coord1.separation(coord2).value


    # In[35]:

    ind=(np.tril(dist_matrix)<1.0/3600.0) & (np.tril(dist_matrix)>0)
    xx,yy=np.meshgrid(np.arange(0,len(prior_list)),np.arange(0,len(prior_list)))
    yy[ind]


    # In[36]:

    prior_list[yy[ind]]


    # In[37]:

    prior_list['redshift'].mask[yy[ind]]=True


    # In[38]:


    prior_list=prior_list[prior_list['redshift'].mask == False]


    # In[39]:

    prior_list


    # XID+ is built around two python classes. A prior and posterior class. There should be a prior class for each map being fitted. It is initiated with a map, noise map, primary header and map header and can be set with a MOC. It also requires an input prior catalogue and point spread function.
    #

    # In[40]:

    #---prior250--------
    prior250=xidplus.prior(im250,nim250,im250phdu,im250hdu, moc=moc)#Initialise with map, uncertianty map, wcs info and primary header
    prior250.prior_cat(prior_list['ra'],prior_list['dec'],'photoz',ID=prior_list['help_id'])
    prior250.prior_bkg(-5.0,5)#Set prior on background (assumes Gaussian pdf with mu and sigma)
    #---prior350--------
    prior350=xidplus.prior(im350,nim350,im350phdu,im350hdu, moc=moc)
    prior350.prior_cat(prior_list['ra'],prior_list['dec'],'photoz',ID=prior_list['help_id'])
    prior350.prior_bkg(-5.0,5)
    #---prior500--------
    prior500=xidplus.prior(im500,nim500,im500phdu,im500hdu, moc=moc)
    prior500.prior_cat(prior_list['ra'],prior_list['dec'],'photoz',ID=prior_list['help_id'])
    prior500.prior_bkg(-5.0,5)


    # Set PSF. For SPIRE, the PSF can be assumed to be Gaussian with a FWHM of 18.15, 25.15, 36.3 '' for 250, 350 and 500 $\mathrm{\mu m}$ respectively. Lets use the astropy module to construct a Gaussian PSF and assign it to the three XID+ prior classes.

    # In[41]:

    #pixsize array (size of pixels in arcseconds)
    pixsize=np.array([pixsize250,pixsize350,pixsize500])
    #point response function for the three bands
    prfsize=np.array([18.15,25.15,36.3])
    #use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
    from astropy.convolution import Gaussian2DKernel

    ##---------fit using Gaussian beam-----------------------
    prf250=Gaussian2DKernel(prfsize[0]/2.355,x_size=101,y_size=101)
    prf250.normalize(mode='peak')
    prf350=Gaussian2DKernel(prfsize[1]/2.355,x_size=101,y_size=101)
    prf350.normalize(mode='peak')
    prf500=Gaussian2DKernel(prfsize[2]/2.355,x_size=101,y_size=101)
    prf500.normalize(mode='peak')

    pind250=np.arange(0,101,1)*1.0/pixsize[0] #get 250 scale in terms of pixel scale of map
    pind350=np.arange(0,101,1)*1.0/pixsize[1] #get 350 scale in terms of pixel scale of map
    pind500=np.arange(0,101,1)*1.0/pixsize[2] #get 500 scale in terms of pixel scale of map

    prior250.set_prf(prf250.array,pind250,pind250)#requires psf as 2d grid, and x and y bins for grid (in pixel scale)
    prior350.set_prf(prf350.array,pind350,pind350)
    prior500.set_prf(prf500.array,pind500,pind500)



    print('fitting '+ str(prior250.nsrc)+' sources \n')
    print('using ' +  str(prior250.snpix)+', '+ str(prior350.snpix)+' and '+ str(prior500.snpix)+' pixels')
    print('source density = {}'.format(prior250.nsrc/moc.area_sq_deg))


    # Before fitting, the prior classes need to take the PSF and calculate how muich each source contributes to each pixel. This process provides what we call a pointing matrix. Lets calculate the pointing matrix for each prior class

    # In[43]:

    prior250.get_pointing_matrix()
    prior350.get_pointing_matrix()
    prior500.get_pointing_matrix()


    # In[44]:

    return [prior250,prior350,prior500],prior_list


def getSEDs(data, src, nsamp=30,category='posterior'):
    import subprocess
    if category=='posterior':
        d=data.posterior
    else:
        d=data.prior

    subsample = np.random.choice(d.chain.size * d.draw.size, size=nsamp,replace=False)

    agn = d.agn.values.reshape(d.chain.size * d.draw.size,
                                            d.src.size)[subsample, :]
    z = d.redshift.values.reshape(d.chain.size * d.draw.size,
                                               d.src.size)[subsample, :]
    sfr = d.sfr.values.reshape(d.chain.size * d.draw.size,
                                            d.src.size)[subsample, :]

    fin = open("/Volumes/pdh_storage/cigale/pcigale_orig.ini")
    fout = open("/Volumes/pdh_storage/cigale/pcigale.ini", "wt")
    for line in fin:
        if 'redshift =' in line:
            fout.write('    redshift = ' + ', '.join(['{:.13f}'.format(i) for i in z[:, src]]) + ' \n')
        elif 'fracAGN =' in line:
            fout.write('    fracAGN = ' + ', '.join(['{:.13f}'.format(i) for i in agn[:, src]]) + ' \n')
        else:
            fout.write(line)
    fin.close()
    fout.close()

    p = subprocess.Popen(['pcigale', 'run'], cwd='/Volumes/pdh_storage/cigale/')
    p.wait()

    SEDs = Table.read('/Volumes/pdh_storage/cigale/out//models-block-0.fits')
    # set more appropriate units for dust
    from astropy.constants import L_sun, M_sun
    SEDs['dust.luminosity'] = SEDs['dust.luminosity'] / L_sun.value
    SEDs['dust.mass'] = SEDs['dust.mass'] / M_sun.value

    wavelengths = []
    fluxes = []
    for i in range(0, nsamp):
        sed_plot = Table.read('/Volumes/pdh_storage/cigale/out/{}_best_model.fits'.format(+SEDs[i * nsamp + (i)]['id']))
        wavelengths.append(sed_plot['wavelength'] / 1E3)
        fluxes.append(((10.0 ** sfr[i, src]) / SEDs[i * nsamp + (i)]['sfh.sfr']) * sed_plot['Fnu'])
    from astropy.table import vstack, hstack

    return hstack(wavelengths), hstack(fluxes)


