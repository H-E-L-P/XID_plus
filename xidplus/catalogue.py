import numpy as np
from astropy.io import fits

def git_version():
    from subprocess import Popen, PIPE
    gitproc = Popen(['git', 'rev-parse','HEAD'], stdout = PIPE)
    (stdout, _) = gitproc.communicate()
    return stdout.strip()

# noinspection PyPackageRequirements
def create_XIDp_SPIREcat_nocov(posterior,prior250,prior350,prior500):
    """creates the XIDp catalogue in fits format required by HeDaM"""
    import datetime
    nsrc=posterior.nsrc
    med_flux=posterior.quantileGet(50)
    flux_low=posterior.quantileGet(15.87)
    flux_high=posterior.quantileGet(84.1)




    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='15A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy', array=med_flux[0:nsrc])
    c5 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy', array=flux_high[0:nsrc])
    c6 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy', array=flux_low[0:nsrc])
    c7 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy', array=med_flux[nsrc+1:(2*nsrc)+1])
    c8 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy', array=flux_high[nsrc+1:(2*nsrc)+1])
    c9 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy', array=flux_low[nsrc+1:(2*nsrc)+1])
    c10 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy', array=med_flux[2*nsrc+2:(3*nsrc)+2])
    c11 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy', array=flux_high[2*nsrc+2:(3*nsrc)+2])
    c12 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy', array=flux_low[2*nsrc+2:(3*nsrc)+2])
    c13 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam', array=np.full(nsrc,med_flux[nsrc]))
    c14 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam', array=np.full(nsrc,med_flux[(2*nsrc)+1]))
    c15 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam', array=np.full(nsrc,med_flux[(3*nsrc)+2]))
    c16 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam', array=np.full(nsrc,med_flux[(3*nsrc)+3]))
    c17 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam', array=np.full(nsrc,med_flux[(3*nsrc)+4]))
    c18 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam', array=np.full(nsrc,med_flux[(3*nsrc)+5]))
    c19 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat[0:nsrc])
    c20 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat[nsrc+1:(2*nsrc)+1])
    c21 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat[2*nsrc+2:(3*nsrc)+2])
    c22 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff[0:nsrc])
    c23 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff[nsrc+1:(2*nsrc)+1])
    c24 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff[2*nsrc+2:(3*nsrc)+2])


    tbhdu = fits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24])
    
    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','250 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','250 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','250 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','350 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','350 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','350 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','500 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','500 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','500 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 background',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','350 background',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','500 background',after='TUCD15')

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')
    tbhdu.header.set('TDESC16','250 residual confusion noise',after='TUCD16')

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')
    tbhdu.header.set('TDESC17','350 residual confusion noise',after='TUCD17')

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')
    tbhdu.header.set('TDESC18','500 residual confusion noise',after='TUCD18')

    tbhdu.header.set('TUCD19','stat.value',after='TFORM16')
    tbhdu.header.set('TDESC19','250 MCMC Convergence statistic',after='TUCD19')

    tbhdu.header.set('TUCD20','stat.value',after='TFORM20')
    tbhdu.header.set('TDESC20','350 MCMC Convergence statistic',after='TUCD20')

    tbhdu.header.set('TUCD21','stat.value',after='TFORM21')
    tbhdu.header.set('TDESC21','500 MCMC Convergence statistic',after='TUCD21')

    tbhdu.header.set('TUCD22','stat.value',after='TFORM22')
    tbhdu.header.set('TDESC22','250 MCMC independence statistic',after='TUCD22')

    tbhdu.header.set('TUCD23','stat.value',after='TFORM23')
    tbhdu.header.set('TDESC23','350 MCMC independence statistic',after='TUCD23')

    tbhdu.header.set('TUCD24','stat.value',after='TFORM24')
    tbhdu.header.set('TDESC24','500 MCMC independence statistic',after='TUCD24')

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat
    prihdr['TITLE']   = 'SPIRE XID+ catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XID+VERSION'] = git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist

def create_empty_XIDp_SPIREcat(nsrc):
    """creates the XIDp catalogue in fits format required by HeDaM"""
    import datetime



    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='I', array=prior250.ID)
    c2 = fits.Column(name='ra', format='D', unit='degrees', array=np.empty((nsrc)))
    c3 = fits.Column(name='dec', format='D', unit='degrees', array=np.empty((nsrc)))
    c4 = fits.Column(name='flux250', format='E', unit='mJy', array=np.empty((nsrc)))
    c5 = fits.Column(name='flux250_err_u', format='E', unit='mJy', array=np.empty((nsrc)))
    c6 = fits.Column(name='flux250_err_l', format='E', unit='mJy', array=np.empty((nsrc)))
    c7 = fits.Column(name='flux350', format='E', unit='mJy', array=np.empty((nsrc)))
    c8 = fits.Column(name='flux350_err_u', format='E', unit='mJy', array=np.empty((nsrc)))
    c9 = fits.Column(name='flux350_err_l', format='E', unit='mJy', array=np.empty((nsrc)))
    c10 = fits.Column(name='flux500', format='E', unit='mJy', array=np.empty((nsrc)))
    c11 = fits.Column(name='flux500_err_u', format='E', unit='mJy', array=np.empty((nsrc)))
    c12 = fits.Column(name='flux500_err_l', format='E', unit='mJy', array=np.empty((nsrc)))
    c13 = fits.Column(name='bkg250', format='E', unit='mJy', array=np.empty((nsrc)))
    c14 = fits.Column(name='bkg350', format='E', unit='mJy', array=np.empty((nsrc)))
    c15 = fits.Column(name='bkg500', format='E', unit='mJy', array=np.empty((nsrc)))

    tbhdu = fits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15])
    
    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

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

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 background',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','350 background',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','500 background',after='TUCD15')
    
    #----Primary header-----------------------------------
    prihdr = fits.Header()
    #prihdr['Prior_C'] = prior250.prior_cat
    prihdr['TITLE']   = 'SPIRE XID catalogue'        
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['VERSION'] = 'beta'                                 
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist

def create_XIDp_SPIREcat_post(posterior,prior250,prior350,prior500):
    """creates the XIDp catalogue in fits format required by HeDaM"""
    import datetime
    nsrc=posterior.nsrc
    med_flux=posterior.quantileGet(50)
    flux_low=posterior.quantileGet(15.87)
    flux_high=posterior.quantileGet(84.1)
    (samp,chains,params)=posterior.stan_fit.shape
    flattened_post=posterior.stan_fit.reshape(samp*chains,params)




    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='A', array=prior250.ID)
    c2 = fits.Column(name='ra', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='flux250_m', format='E', unit='mJy', array=med_flux[0:nsrc])
    c5 = fits.Column(name='flux250_u', format='E', unit='mJy', array=flux_high[0:nsrc])
    c6 = fits.Column(name='flux250_l', format='E', unit='mJy', array=flux_low[0:nsrc])
    c7 = fits.Column(name='flux350_m', format='E', unit='mJy', array=med_flux[nsrc+1:(2*nsrc)+1])
    c8 = fits.Column(name='flux350_u', format='E', unit='mJy', array=flux_high[nsrc+1:(2*nsrc)+1])
    c9 = fits.Column(name='flux350_l', format='E', unit='mJy', array=flux_low[nsrc+1:(2*nsrc)+1])
    c10 = fits.Column(name='flux500_m', format='E', unit='mJy', array=med_flux[2*nsrc+2:(3*nsrc)+2])
    c11 = fits.Column(name='flux500_u', format='E', unit='mJy', array=flux_high[2*nsrc+2:(3*nsrc)+2])
    c12 = fits.Column(name='flux500_l', format='E', unit='mJy', array=flux_low[2*nsrc+2:(3*nsrc)+2])
    c13 = fits.Column(name='bkg250_m', format='E', unit='mJy', array=np.full(nsrc,med_flux[nsrc]))
    c14 = fits.Column(name='bkg350_m', format='E', unit='mJy', array=np.full(nsrc,med_flux[(2*nsrc)+1]))
    c15 = fits.Column(name='bkg500_m', format='E', unit='mJy', array=np.full(nsrc,med_flux[(3*nsrc)+2]))
    c16 = fits.Column(name='Rhat_250', format='E', array=posterior.Rhat[0:nsrc])
    c17 = fits.Column(name='Rhat_350', format='E', array=posterior.Rhat[nsrc+1:(2*nsrc)+1])
    c18 = fits.Column(name='Rhat_500', format='E', array=posterior.Rhat[2*nsrc+2:(3*nsrc)+2])
    c19 = fits.Column(name='n_eff_250', format='E', array=posterior.n_eff[0:nsrc])
    c20 = fits.Column(name='n_eff_350', format='E', array=posterior.n_eff[nsrc+1:(2*nsrc)+1])
    c21 = fits.Column(name='n_eff_500', format='E', array=posterior.n_eff[2*nsrc+2:(3*nsrc)+2])
    c22 = fits.Column(name='post_250', format=str(samp*chains)+'E', array=flattened_post[:,0:nsrc].T)
    c23 = fits.Column(name='post_350', format=str(samp*chains)+'E', array=flattened_post[:,nsrc+1:(2*nsrc)+1].T)
    c24 = fits.Column(name='post_500', format=str(samp*chains)+'E', array=flattened_post[:,2*nsrc+2:(3*nsrc)+2].T)
    c25 = fits.Column(name='post_bkg250', format=str(samp*chains)+'E', unit='mJy', array=np.tile(flattened_post[:,nsrc],(nsrc,1)))
    c26 = fits.Column(name='post_bkg350', format=str(samp*chains)+'E', unit='mJy', array=np.tile(flattened_post[:,(2*nsrc)+1],(nsrc,1)))
    c27 = fits.Column(name='post_bkg500', format=str(samp*chains)+'E', unit='mJy', array=np.tile(flattened_post[:,(3*nsrc)+2],(nsrc,1)))

    tbhdu = fits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27])
    
    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')      
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2') 

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')      
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3') 

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')      
    tbhdu.header.set('TDESC4','250 Flux (at 50th percentile)',after='TUCD4') 

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')      
    tbhdu.header.set('TDESC5','250 Flux (at 84.1 percentile) ',after='TUCD5') 

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')      
    tbhdu.header.set('TDESC6','250 Flux (at 15.9 percentile)',after='TUCD6') 

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')      
    tbhdu.header.set('TDESC7','350 Flux (at 50th percentile)',after='TUCD7') 

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')      
    tbhdu.header.set('TDESC8','350 Flux (at 84.1 percentile) ',after='TUCD8') 

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')      
    tbhdu.header.set('TDESC9','350 Flux (at 15.9 percentile)',after='TUCD9') 

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')      
    tbhdu.header.set('TDESC10','500 Flux (at 50th percentile)',after='TUCD10') 

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')      
    tbhdu.header.set('TDESC11','500 Flux (at 84.1 percentile) ',after='TUCD11') 

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')      
    tbhdu.header.set('TDESC12','500 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')      
    tbhdu.header.set('TDESC13','250 background',after='TUCD13') 

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')      
    tbhdu.header.set('TDESC14','350 background',after='TUCD14') 

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')      
    tbhdu.header.set('TDESC15','500 background',after='TUCD15')

    tbhdu.header.set('TUCD16','stat.value',after='TFORM16')      
    tbhdu.header.set('TDESC16','250 MCMC Convergence statistic',after='TUCD16')

    tbhdu.header.set('TUCD17','stat.value',after='TFORM17')      
    tbhdu.header.set('TDESC17','350 MCMC Convergence statistic',after='TUCD17')

    tbhdu.header.set('TUCD18','stat.value',after='TFORM18')      
    tbhdu.header.set('TDESC18','500 MCMC Convergence statistic',after='TUCD18')

    tbhdu.header.set('TUCD19','stat.value',after='TFORM19')      
    tbhdu.header.set('TDESC19','250 MCMC independence statistic',after='TUCD19')

    tbhdu.header.set('TUCD20','stat.value',after='TFORM20')      
    tbhdu.header.set('TDESC20','350 MCMC independence statistic',after='TUCD20')

    tbhdu.header.set('TUCD21','stat.value',after='TFORM21')      
    tbhdu.header.set('TDESC21','500 MCMC independence statistic',after='TUCD21')

    tbhdu.header.set('TUCD22','phot.flux.density',after='TFORM22')      
    tbhdu.header.set('TDESC22','250 samples',after='TUCD22')

    tbhdu.header.set('TUCD23','phot.flux.density',after='TFORM23')      
    tbhdu.header.set('TDESC23','350 samples',after='TUCD23')
    
    tbhdu.header.set('TUCD24','phot.flux.density',after='TFORM24')      
    tbhdu.header.set('TDESC24','500 samples',after='TUCD24')

    tbhdu.header.set('TUCD25','phot.flux.density',after='TFORM25')      
    tbhdu.header.set('TDESC25','250 bkg samples',after='TUCD25')

    tbhdu.header.set('TUCD26','phot.flux.density',after='TFORM26')      
    tbhdu.header.set('TDESC26','350 bkg samples',after='TUCD26')
    
    tbhdu.header.set('TUCD27','phot.flux.density',after='TFORM27')      
    tbhdu.header.set('TDESC27','500 bkg samples',after='TUCD27')



    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_C'] = prior250.prior_cat
    prihdr['TITLE']   = 'SPIRE XID catalogue'        
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['VERSION'] = 'beta'                                 
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist

# noinspection PyPackageRequirements
def create_XIDp_SPIREcat_sample(posterior,prior250,prior350,prior500):
    """creates the XIDp catalogue in fits format required by HeDaM"""
    import datetime
    nsrc=posterior.nsrc
    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='ID', format='15A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='flux250', format='E', unit='mJy', array=med_flux[0:nsrc])

    c5 = fits.Column(name='flux350', format='E', unit='mJy', array=med_flux[nsrc+1:(2*nsrc)+1])

    c6 = fits.Column(name='flux500', format='E', unit='mJy', array=med_flux[2*nsrc+2:(3*nsrc)+2])

    c7 = fits.Column(name='bkg250', format='E', unit='mJy', array=np.full(nsrc,med_flux[nsrc]))
    c8 = fits.Column(name='bkg350', format='E', unit='mJy', array=np.full(nsrc,med_flux[(2*nsrc)+1]))
    c9 = fits.Column(name='bkg500', format='E', unit='mJy', array=np.full(nsrc,med_flux[(3*nsrc)+2]))
    c10 = fits.Column(name='sig_conf250', format='E', array=np.full(nsrc,med_flux[(3*nsrc)+2]))
    c11 = fits.Column(name='sig_conf350', format='E', array=np.full(nsrc,med_flux[(3*nsrc)+3]))
    c12 = fits.Column(name='sig_conf500', format='E', array=np.full(nsrc,med_flux[(3*nsrc)+4]))
    c13 = fits.Column(name='Rhat_250', format='E', array=posterior.Rhat[0:nsrc])
    c14 = fits.Column(name='Rhat_350', format='E', array=posterior.Rhat[nsrc+1:(2*nsrc)+1])
    c15 = fits.Column(name='Rhat_500', format='E', array=posterior.Rhat[2*nsrc+2:(3*nsrc)+2])
    c16 = fits.Column(name='n_eff_250', format='E', array=posterior.n_eff[0:nsrc])
    c17 = fits.Column(name='n_eff_350', format='E', array=posterior.n_eff[nsrc+1:(2*nsrc)+1])
    c18 = fits.Column(name='n_eff_500', format='E', array=posterior.n_eff[2*nsrc+2:(3*nsrc)+2])


    tbhdu = fits.new_table([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18])

    tbhdu.header.set('TUCD1','ID',after='TFORM1')
    tbhdu.header.set('TDESC1','ID of source',after='TUCD1')

    tbhdu.header.set('TUCD2','pos.eq.RA',after='TUNIT2')
    tbhdu.header.set('TDESC2','R.A. of object J2000',after='TUCD2')
    tbhdu.header.set('TUNIT2','deg.',after='TUCD2')

    tbhdu.header.set('TUCD3','pos.eq.DEC',after='TUNIT3')
    tbhdu.header.set('TDESC3','Dec. of object J2000',after='TUCD3')

    tbhdu.header.set('TUCD4','phot.flux.density',after='TUNIT4')
    tbhdu.header.set('TDESC4','250 Flux (at 50th percentile)',after='TUCD4')

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')
    tbhdu.header.set('TDESC5','250 Flux (at 84.1 percentile) ',after='TUCD5')

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')
    tbhdu.header.set('TDESC6','250 Flux (at 15.9 percentile)',after='TUCD6')

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')
    tbhdu.header.set('TDESC7','350 Flux (at 50th percentile)',after='TUCD7')

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')
    tbhdu.header.set('TDESC8','350 Flux (at 84.1 percentile) ',after='TUCD8')

    tbhdu.header.set('TUCD9','phot.flux.density',after='TUNIT9')
    tbhdu.header.set('TDESC9','350 Flux (at 15.9 percentile)',after='TUCD9')

    tbhdu.header.set('TUCD10','phot.flux.density',after='TUNIT10')
    tbhdu.header.set('TDESC10','500 Flux (at 50th percentile)',after='TUCD10')

    tbhdu.header.set('TUCD11','phot.flux.density',after='TUNIT11')
    tbhdu.header.set('TDESC11','500 Flux (at 84.1 percentile) ',after='TUCD11')

    tbhdu.header.set('TUCD12','phot.flux.density',after='TUNIT12')
    tbhdu.header.set('TDESC12','500 Flux (at 15.9 percentile)',after='TUCD12')

    tbhdu.header.set('TUCD13','phot.flux.density',after='TUNIT13')
    tbhdu.header.set('TDESC13','250 background',after='TUCD13')

    tbhdu.header.set('TUCD14','phot.flux.density',after='TUNIT14')
    tbhdu.header.set('TDESC14','350 background',after='TUCD14')

    tbhdu.header.set('TUCD15','phot.flux.density',after='TUNIT15')
    tbhdu.header.set('TDESC15','500 background',after='TUCD15')

    tbhdu.header.set('TUCD16','phot.flux.density',after='TUNIT16')
    tbhdu.header.set('TDESC16','250 residual confusion noise',after='TUCD16')

    tbhdu.header.set('TUCD17','phot.flux.density',after='TUNIT17')
    tbhdu.header.set('TDESC17','350 residual confusion noise',after='TUCD17')

    tbhdu.header.set('TUCD18','phot.flux.density',after='TUNIT18')
    tbhdu.header.set('TDESC18','500 residual confusion noise',after='TUCD18')

    tbhdu.header.set('TUCD19','stat.value',after='TFORM16')
    tbhdu.header.set('TDESC19','250 MCMC Convergence statistic',after='TUCD19')

    tbhdu.header.set('TUCD20','stat.value',after='TFORM20')
    tbhdu.header.set('TDESC20','350 MCMC Convergence statistic',after='TUCD20')

    tbhdu.header.set('TUCD21','stat.value',after='TFORM21')
    tbhdu.header.set('TDESC21','500 MCMC Convergence statistic',after='TUCD21')

    tbhdu.header.set('TUCD22','stat.value',after='TFORM22')
    tbhdu.header.set('TDESC22','250 MCMC independence statistic',after='TUCD22')

    tbhdu.header.set('TUCD23','stat.value',after='TFORM23')
    tbhdu.header.set('TDESC23','350 MCMC independence statistic',after='TUCD23')

    tbhdu.header.set('TUCD24','stat.value',after='TFORM24')
    tbhdu.header.set('TDESC24','500 MCMC independence statistic',after='TUCD24')

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_C'] = prior250.prior_cat
    prihdr['TITLE']   = 'SPIRE XID+ catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change
    prihdr['CREATOR'] = 'WP5'
    prihdr['XID+VERSION'] = git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()
    prihdu = fits.PrimaryHDU(header=prihdr)

    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist