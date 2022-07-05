import numpy as np
from astropy.io import fits
import xidplus.io as io
import xidplus.posterior_maps as postmaps




def create_PACS_cat(posterior, prior100, prior160):

    """
    Create PACS catalogue from posterior
    
    :param posterior: PACS xidplus.posterior class
    :param prior100:  PACS 100 xidplus.prior class
    :param prior160:  PACS 160 xidplus.prior class
    :return: fits hdulist
    """
    import datetime
    nsrc=prior100.nsrc
    rep_maps=postmaps.replicated_maps([prior100,prior160],posterior)
    Bayes_P100=postmaps.Bayes_Pval_res(prior100,rep_maps[0])
    Bayes_P160=postmaps.Bayes_Pval_res(prior160,rep_maps[1])


    # ----table info-----------------------
    # first define columns
    c1 = fits.Column(name='help_id', format='27A', array=prior100.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior100.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior100.sdec)
    c4 = fits.Column(name='F_PACS_100', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_PACS_100_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_PACS_100_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_PACS_160', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_PACS_160_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_PACS_160_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='Bkg_PACS_100', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c11 = fits.Column(name='Bkg_PACS_160', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c12 = fits.Column(name='Sig_conf_PACS_100', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c13 = fits.Column(name='Sig_conf_PACS_160', format='E', unit='mJy/Beam',
                      array=np.full(nsrc, np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c14 = fits.Column(name='Rhat_PACS_100', format='E', array=posterior.Rhat['src_f'][:,0])
    c15 = fits.Column(name='Rhat_PACS_160', format='E', array=posterior.Rhat['src_f'][:,1])
    c16 = fits.Column(name='n_eff_PACS_100', format='E', array=posterior.n_eff['src_f'][:,0])
    c17 = fits.Column(name='n_eff_PACS_160', format='E', array=posterior.n_eff['src_f'][:,1])
    c18 = fits.Column(name='Pval_res_100', format='E', array=Bayes_P100)
    c19 = fits.Column(name='Pval_res_160', format='E', array=Bayes_P160)


    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19])

    tbhdu.header.set('TUCD1', 'ID', after='TFORM1')
    tbhdu.header.set('TDESC1', 'ID of source', after='TUCD1')

    tbhdu.header.set('TUCD2', 'pos.eq.RA', after='TUNIT2')
    tbhdu.header.set('TDESC2', 'R.A. of object J2000', after='TUCD2')

    tbhdu.header.set('TUCD3', 'pos.eq.DEC', after='TUNIT3')
    tbhdu.header.set('TDESC3', 'Dec. of object J2000', after='TUCD3')

    tbhdu.header.set('TUCD4', 'phot.flux.density', after='TUNIT4')
    tbhdu.header.set('TDESC4', '100 Flux (at 50th percentile)', after='TUCD4')

    tbhdu.header.set('TUCD5', 'phot.flux.density', after='TUNIT5')
    tbhdu.header.set('TDESC5', '100 Flux (at 84.1 percentile) ', after='TUCD5')

    tbhdu.header.set('TUCD6', 'phot.flux.density', after='TUNIT6')
    tbhdu.header.set('TDESC6', '100 Flux (at 15.9 percentile)', after='TUCD6')

    tbhdu.header.set('TUCD7', 'phot.flux.density', after='TUNIT7')
    tbhdu.header.set('TDESC7', '160 Flux (at 50th percentile)', after='TUCD7')

    tbhdu.header.set('TUCD8', 'phot.flux.density', after='TUNIT8')
    tbhdu.header.set('TDESC8', '160 Flux (at 84.1 percentile) ', after='TUCD8')

    tbhdu.header.set('TUCD9', 'phot.flux.density', after='TUNIT9')
    tbhdu.header.set('TDESC9', '160 Flux (at 15.9 percentile)', after='TUCD9')

    tbhdu.header.set('TUCD10', 'phot.flux.density', after='TUNIT10')
    tbhdu.header.set('TDESC10', '100 background', after='TUCD10')

    tbhdu.header.set('TUCD11', 'phot.flux.density', after='TUNIT11')
    tbhdu.header.set('TDESC11', '160 background', after='TUCD11')

    tbhdu.header.set('TUCD12', 'phot.flux.density', after='TUNIT12')
    tbhdu.header.set('TDESC12', '100 residual confusion noise', after='TUCD12')

    tbhdu.header.set('TUCD13', 'phot.flux.density', after='TUNIT13')
    tbhdu.header.set('TDESC13', '160 residual confusion noise', after='TUCD13')

    tbhdu.header.set('TUCD14', 'stat.value', after='TFORM14')
    tbhdu.header.set('TDESC14', '100 MCMC Convergence statistic', after='TUCD14')

    tbhdu.header.set('TUCD15', 'stat.value', after='TFORM15')
    tbhdu.header.set('TDESC15', '160 MCMC Convergence statistic', after='TUCD15')

    tbhdu.header.set('TUCD16', 'stat.value', after='TFORM16')
    tbhdu.header.set('TDESC16', '100 MCMC independence statistic', after='TUCD16')

    tbhdu.header.set('TUCD17', 'stat.value', after='TFORM17')
    tbhdu.header.set('TDESC17', '160 MCMC independence statistic', after='TUCD17')
    
    tbhdu.header.set('TUCD18','stat.value',after='TFORM18')
    tbhdu.header.set('TDESC18','100 Bayes Pval residual statistic',after='TUCD18')

    tbhdu.header.set('TUCD19','stat.value',after='TFORM19')
    tbhdu.header.set('TDESC19','160 Bayes Pval residual statistic',after='TUCD19')
    # ----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior100.prior_cat_file
    prihdr['TITLE'] = 'PACS XID+ catalogue'
    # prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change
    prihdr['CREATOR'] = 'WP5'
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE'] = datetime.datetime.now().isoformat()
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist

# noinspection PyPackageRequirements

def create_MIPS_cat(posterior, prior24, Bayes_P24):

    """
    Create MIPS catalogue from posterior
    
    :param posterior: MIPS xidplus.posterior class
    :param prior24: MIPS xidplus.prior class
    :param Bayes_P24:  Bayes Pvalue residual statistic for MIPS 24
    :return: fits hdulist
    """
    import datetime
    nsrc=prior24.nsrc
    rep_maps = postmaps.replicated_maps([prior24], posterior)
    Bayes_P24 = postmaps.Bayes_Pval_res(prior24, rep_maps[0])
    # ----table info-----------------------
    # first define columns
    c1 = fits.Column(name='help_id', format='27A', array=prior24.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior24.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior24.sdec)
    c4 = fits.Column(name='F_MIPS_24', format='E', unit='muJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_MIPS_24_u', format='E', unit='muJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_MIPS_24_l', format='E', unit='muJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='Bkg_MIPS_24', format='E', unit='MJy/sr',
                     array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c8 = fits.Column(name='Sig_conf_MIPS_24', format='E', unit='MJy/sr',
                     array=np.full(nsrc, np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c9 = fits.Column(name='Rhat_MIPS_24', format='E', array=posterior.Rhat['src_f'][:,0])
    c10 = fits.Column(name='n_eff_MIPS_24', format='E', array=posterior.n_eff['src_f'][:,0])
    c11 = fits.Column(name='Pval_res_24', format='E', array=Bayes_P24)
    
    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])

    tbhdu.header.set('TUCD1', 'ID', after='TFORM1')
    tbhdu.header.set('TDESC1', 'ID of source', after='TUCD1')

    tbhdu.header.set('TUCD2', 'pos.eq.RA', after='TUNIT2')
    tbhdu.header.set('TDESC2', 'R.A. of object J2000', after='TUCD2')

    tbhdu.header.set('TUCD3', 'pos.eq.DEC', after='TUNIT3')
    tbhdu.header.set('TDESC3', 'Dec. of object J2000', after='TUCD3')

    tbhdu.header.set('TUCD4', 'phot.flux.density', after='TUNIT4')
    tbhdu.header.set('TDESC4', '24 Flux (at 50th percentile)', after='TUCD4')

    tbhdu.header.set('TUCD5','phot.flux.density',after='TUNIT5')
    tbhdu.header.set('TDESC5','24 Flux (at 84.1 percentile) ',after='TUCD5')

    tbhdu.header.set('TUCD6','phot.flux.density',after='TUNIT6')
    tbhdu.header.set('TDESC6','24 Flux (at 15.9 percentile)',after='TUCD6')

    tbhdu.header.set('TUCD7','phot.flux.density',after='TUNIT7')
    tbhdu.header.set('TDESC7','24 background',after='TUCD7')

    tbhdu.header.set('TUCD8','phot.flux.density',after='TUNIT8')
    tbhdu.header.set('TDESC8','24 residual confusion noise',after='TUCD8')

    tbhdu.header.set('TUCD9','stat.value',after='TFORM9')
    tbhdu.header.set('TDESC9','24 MCMC Convergence statistic',after='TUCD9')

    tbhdu.header.set('TUCD10','stat.value',after='TFORM10')
    tbhdu.header.set('TDESC10','24 MCMC independence statistic',after='TUCD10')

    tbhdu.header.set('TUCD11','stat.value',after='TFORM11')
    tbhdu.header.set('TDESC11','24 Bayes Pval residual statistic',after='TUCD11')

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior24.prior_cat_file
    prihdr['TITLE']   = 'XID+MIPS catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change
    prihdr['CREATOR'] = 'WP5'
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist
# noinspection PyPackageRequirements


def create_SPIRE_cat(posterior,prior250,prior350,prior500):

    """
    Create SPIRE catalogue from posterior


    :param posterior: SPIRE xidplus.posterior class
    :param prior250: SPIRE 250 xidplus.prior class
    :param prior350: SPIRE 350 xidplus.prior class
    :param prior500: SPIRE 500 xidplus.prior class
    :param Bayes_P250: Bayes Pvalue residual statistic for SPIRE 250
    :param Bayes_P350: Bayes Pvalue residual statistic for SPIRE 350
    :param Bayes_P500: Bayes Pvalue residual statistic for SPIRE 500
    :return: fits hdulist
    """
    import datetime
    nsrc=posterior.nsrc
    rep_maps = postmaps.replicated_maps([prior250, prior350,prior500], posterior)
    Bayes_P250 = postmaps.Bayes_Pval_res(prior250, rep_maps[0])
    Bayes_P350 = postmaps.Bayes_Pval_res(prior350, rep_maps[1])
    Bayes_P500 = postmaps.Bayes_Pval_res(prior500, rep_maps[2])


    #----table info-----------------------
    #first define columns
    c1 = fits.Column(name='HELP_ID', format='27A', array=prior250.ID)
    c2 = fits.Column(name='RA', format='D', unit='degrees', array=prior250.sra)
    c3 = fits.Column(name='Dec', format='D', unit='degrees', array=prior250.sdec)
    c4 = fits.Column(name='F_SPIRE_250', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],50.0,axis=0))
    c5 = fits.Column(name='FErr_SPIRE_250_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],84.1,axis=0))
    c6 = fits.Column(name='FErr_SPIRE_250_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,0,:],15.9,axis=0))
    c7 = fits.Column(name='F_SPIRE_350', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],50.0,axis=0))
    c8 = fits.Column(name='FErr_SPIRE_350_u', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],84.1,axis=0))
    c9 = fits.Column(name='FErr_SPIRE_350_l', format='E', unit='mJy',
                     array=np.percentile(posterior.samples['src_f'][:,1,:],15.9,axis=0))
    c10 = fits.Column(name='F_SPIRE_500', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],50.0,axis=0))
    c11 = fits.Column(name='FErr_SPIRE_500_u', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],84.1,axis=0))
    c12 = fits.Column(name='FErr_SPIRE_500_l', format='E', unit='mJy',
                      array=np.percentile(posterior.samples['src_f'][:,2,:],15.9,axis=0))
    c13 = fits.Column(name='Bkg_SPIRE_250', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,0],50.0,axis=0)))
    c14 = fits.Column(name='Bkg_SPIRE_350', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,1],50.0,axis=0)))
    c15 = fits.Column(name='Bkg_SPIRE_500', format='E', unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['bkg'][:,2],50.0,axis=0)))
    c16 = fits.Column(name='Sig_conf_SPIRE_250', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,0],50.0,axis=0)))
    c17 = fits.Column(name='Sig_conf_SPIRE_350', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,1],50.0,axis=0)))
    c18 = fits.Column(name='Sig_conf_SPIRE_500', format='E',unit='mJy/Beam',
                      array=np.full(nsrc,np.percentile(posterior.samples['sigma_conf'][:,2],50.0,axis=0)))
    c19 = fits.Column(name='Rhat_SPIRE_250', format='E', array=posterior.Rhat['src_f'][:,0])
    c20 = fits.Column(name='Rhat_SPIRE_350', format='E', array=posterior.Rhat['src_f'][:,1])
    c21 = fits.Column(name='Rhat_SPIRE_500', format='E', array=posterior.Rhat['src_f'][:,2])
    c22 = fits.Column(name='n_eff_SPIRE_250', format='E', array=posterior.n_eff['src_f'][:,0])
    c23 = fits.Column(name='n_eff_SPIRE_350', format='E', array=posterior.n_eff['src_f'][:,1])
    c24 = fits.Column(name='n_eff_SPIRE_500', format='E', array=posterior.n_eff['src_f'][:,2])
    c25 = fits.Column(name='Pval_res_250', format='E', array=Bayes_P250)
    c26 = fits.Column(name='Pval_res_350', format='E', array=Bayes_P350)
    c27 = fits.Column(name='Pval_res_500', format='E', array=Bayes_P500)
    

    tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,
                            c24, c23, c25, c26, c27])

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

    tbhdu.header.set('TUCD25','stat.value',after='TFORM25')
    tbhdu.header.set('TDESC25','250 Bayes Pval residual statistic',after='TUCD25')

    tbhdu.header.set('TUCD26','stat.value',after='TFORM26')
    tbhdu.header.set('TDESC26','350 Bayes Pval residual statistic',after='TUCD26')

    tbhdu.header.set('TUCD27','stat.value',after='TFORM27')
    tbhdu.header.set('TDESC27','500 Bayes Pval residual statistic',after='TUCD27')

    #----Primary header-----------------------------------
    prihdr = fits.Header()
    prihdr['Prior_Cat'] = prior250.prior_cat_file
    prihdr['TITLE']   = 'SPIRE XID+ catalogue'
    #prihdr['OBJECT']  = prior250.imphdu['OBJECT'] #I need to think if this needs to change                              
    prihdr['CREATOR'] = 'WP5'                                 
    prihdr['XIDplus'] = io.git_version()
    prihdr['DATE']    = datetime.datetime.now().isoformat()              
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    thdulist = fits.HDUList([prihdu, tbhdu])
    return thdulist

