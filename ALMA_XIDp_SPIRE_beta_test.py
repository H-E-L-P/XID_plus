import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs
import pickle
import dill
import XIDp_mod_beta as xid_mod
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS
from astropy.coordinates import Angle

#Get field names out of ALMA data catalogue:
fields=[]
sources=[]
ralist=[]
declist=[]
file=open('/Users/jillianscudder/Research/ALMA_XID/table_observed.dat', 'r')
dat=file.readlines()
for entry in dat[1::]:
    splitlines=entry.split()
    fields.append(splitlines[0])
    sources.append(splitlines[1])
    ralist.append(splitlines[2])
    declist.append(splitlines[3])
file.close()
fields=np.asarray(fields)
sources=np.asarray(sources)
ralist=np.asarray(ralist)
declist=np.asarray(declist)

#Find the unique field IDs
unique_fields=np.unique(fields)
print 'Number of fields:', len(unique_fields)
#Select the number of sources per field
sources_per_field={}
ra_per_field={}
dec_per_field={}
for u_field in unique_fields:
    sources_per_field[u_field]=sources[np.where(fields==u_field)]
    ra_per_field[u_field]=ralist[np.where(fields==u_field)]
    dec_per_field[u_field]=declist[np.where(fields==u_field)]

#Convert RA/Dec lists into decimal from hours
field_ra_dict={}
field_dec_dict={}
for field in unique_fields:
    print field
    sources=sources_per_field[field]
    ratemp=ra_per_field[field]
    dectemp=dec_per_field[field]
    ra_decdeg_dict={}
    dec_decdeg_dict={}
    for index, entry in enumerate(ratemp):
        source=sources[index]
        coord_temp=SkyCoord(entry, dectemp[index], unit=(u.hourangle, u.deg))
        print coord_temp.ra.deg
        print coord_temp.dec.deg
        #dec_temp=SkyCoord(dectemp[index], unit=u.deg)
        #print (ra_temp.to_string(unit='degree', decimal='True', precision=7)), (dec_temp.to_string(unit='degree', decimal='True', precision=7))
        ra_decdeg_dict[source]=(coord_temp.ra.to_string(unit='degree', decimal='True', precision=10))
        dec_decdeg_dict[source]=(coord_temp.dec.to_string(unit='degree', decimal='True', precision=10))
    field_ra_dict[field]=ra_decdeg_dict
    field_dec_dict[field]=dec_decdeg_dict
    print 'end', field

#Folder containing maps
imfolder='/Users/jillianscudder/Research/ALMA_XID/'
#field
for index in range(0, len(unique_fields)):
    print "____________________________"
    print "Currently fitting field: ", unique_fields[index]
    field=unique_fields[index]

    pswfits=imfolder+'SPIRE250/'+field+'_250.fits'#SPIRE 250 map
    pmwfits=imfolder+'SPIRE350/'+field+'_350.fits'#SPIRE 350 map
    plwfits=imfolder+'SPIRE500/'+field+'_500.fits'#SPIRE 500 map
    ##My prior catalogue is the 870 source positions, already read in
    prior_cat="870 positions"
    # Open images and noise maps and use WCS module in astropy to get header information
    # In[9]:

    #-----250-------------
    hdulist = fits.open(pswfits, ignore_missing_end=True)
    im250phdu=hdulist[0].header
    im250=(hdulist[1].data*1.0E3) #Convert to mJy
    nim250=hdulist[2].data*1.0E3
    w_250 = wcs.WCS(hdulist[1].header)
    pixsize250=3600.0*w_250.wcs.cd[1,1] #pixel size (in arcseconds)
    print 'Arcseconds per pixel, 250 micron:', pixsize250
    hdulist.close()
    #-----350-------------
    hdulist = fits.open(pmwfits, ignore_missing_end=True)
    im350phdu=hdulist[0].header
    im350=hdulist[1].data*1.0E3
    nim350=hdulist[2].data*1.0E3
    w_350 = wcs.WCS(hdulist[1].header)
    pixsize350=3600.0*w_350.wcs.cd[1,1] #pixel size (in arcseconds)
    print 'Arcseconds per pixel, 350 micron:', pixsize350
    hdulist.close()
    #-----500-------------
    hdulist = fits.open(plwfits, ignore_missing_end=True)
    im500phdu=hdulist[0].header
    im500=hdulist[1].data*1.0E3
    nim500=hdulist[2].data*1.0E3
    w_500 = wcs.WCS(hdulist[1].header)
    pixsize500=3600.0*w_500.wcs.cd[1,1] #pixel size (in arcseconds)
    print 'Arcseconds per pixel, 500 micron:', pixsize500
    hdulist.close()

    # In[12]:
    #Count number of sources per object:
    fieldid=field
    sourcelist=sources_per_field[fieldid]
    print sourcelist
    inra=[]
    indec=[]
    for gal in sourcelist:
        inra.append(float(field_ra_dict[field][gal]))
        indec.append(float(field_dec_dict[field][gal]))
    
    print 'Fitting '+str(len(inra))+' sources...'

    inra=np.asarray(inra)
    indec=np.asarray(indec)

    # Point response information, at the moment its 2D Gaussian,

    #pixsize array (size of pixels in arcseconds)
    pixsize=np.array([pixsize250,pixsize350,pixsize500])
    #point response function for the three bands
    prfsize=np.array([18.15,25.15,36.3])
    #use Gaussian2DKernel to create prf (requires stddev rather than fwhm hence pfwhm/2.355)
    from astropy.convolution import Gaussian2DKernel



    prior250=xid_mod.prior(im250,nim250,w_250,im250phdu)
    prior250.prior_cat(inra,indec,prior_cat)
    prior250.prior_bkg(0,2)
    prior350=xid_mod.prior(im350,nim350,w_350,im350phdu)
    prior350.prior_cat(inra,indec,prior_cat)
    prior350.prior_bkg(0,2)
    prior500=xid_mod.prior(im500,nim500,w_500,im500phdu)
    prior500.prior_cat(inra,indec,prior_cat)
    prior500.prior_bkg(0,2)

    #thdulist,prior250,prior350,prior500,posterior=xid_mod.fit_SPIRE(prior250,prior350,prior500)

    #-----------fit using real beam--------------------------
    #PSF_250,px_250,py_250=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PSW_bgmod9_1arcsec.fits',pixsize[0])
    #PSF_350,px_350,py_350=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PMW_bgmod9_1arcsec.fits',pixsize[1])
    #PSF_500,px_500,py_500=xid_mod.SPIRE_PSF('../hsc-calibration/0x5000241aL_PLW_bgmod9_1arcsec.fits',pixsize[2])
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

    prior250.set_prf(prf250.array,pind250,pind250)
    prior350.set_prf(prf350.array,pind350,pind350)
    prior500.set_prf(prf500.array,pind500,pind500)

    prior250.get_pointing_matrix()
    prior350.get_pointing_matrix()
    prior500.get_pointing_matrix()

    fit_data,chains,iter=xid_mod.lstdrv_SPIRE_stan(prior250,prior350,prior500)
    posterior=xid_mod.posterior_stan(fit_data[:,:,0:-1],prior250.nsrc)
    thdulist=xid_mod.create_XIDp_SPIREcat(posterior,prior250,prior350,prior500)
    #----------------------------------------------------------
    
    
    output_folder='/Users/jillianscudder/XID_plus/Output/870MN/'
    thdulist.writeto(output_folder+'XIDp_ALMA_SPIRE_beta_'+field+'_dat.fits')
    outfile=output_folder+'XIDp_SPIRE_beta_test'+field+'.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump({'psw':prior250,'pmw':prior350,'plw':prior500, 'post':posterior},f)

