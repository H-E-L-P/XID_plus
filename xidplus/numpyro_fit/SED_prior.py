_author__ = 'pdh21'

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
import jax.numpy as jnp
import numpy as np
from xidplus.numpyro_fit.misc import sp_matmul



def spire_model_CIGALE(priors, sed_prior,params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    #get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    #background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # redshift-sfr relation parameters
    m = numpyro.sample('m', dist.Normal(params['m_mu'], params['m_sig']))
    c = numpyro.sample('c', dist.Normal(params['c_mu'], params['c_sig']))

    #sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    #sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        #use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(0.01, sed_prior.params_mu[:, 1], sed_prior.params_sig[:, 1]))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        # use handlers.reparam as sampling from standard normal and transforming by redshift relation
        #with numpyro.handlers.reparam(config={"params": TransformReparam()}):
            #sfr = numpyro.sample('sfr', dist.TransformedDistribution(dist.Normal(0.0, 1.0),
                                                                     #dist.transforms.AffineTransform(redshift * m + c,
                                                                                                    # jnp.full(
                                                                                                        # priors[0].nsrc,
                                                                                                        # sfr_sig))))
        sfr = numpyro.sample('sfr', dist.Normal(redshift * m + c,jnp.full(priors[0].nsrc,sfr_sig)))

    #stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_plw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]

    # for each band, condition on data
    with numpyro.plate('psw_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('plw_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[2].snim),
                       obs=priors[2].sim)


def spire_model_CIGALE(priors, sed_prior, params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # redshift-sfr relation parameters
    m = numpyro.sample('m', dist.Normal(params['m_mu'], params['m_sig']))
    c = numpyro.sample('c', dist.Normal(params['c_mu'], params['c_sig']))

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(0.01, sed_prior.params_mu[:, 1], sed_prior.params_sig[:, 1]))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        # use handlers.reparam as sampling from standard normal and transforming by redshift relation
        # with numpyro.handlers.reparam(config={"params": TransformReparam()}):
        # sfr = numpyro.sample('sfr', dist.TransformedDistribution(dist.Normal(0.0, 1.0),
        # dist.transforms.AffineTransform(redshift * m + c,
        # jnp.full(
        # priors[0].nsrc,
        # sfr_sig))))
        sfr = numpyro.sample('sfr', dist.Normal(redshift * m + c, jnp.full(priors[0].nsrc, sfr_sig)))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_plw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]

    # for each band, condition on data
    with numpyro.plate('psw_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('plw_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[2].snim),
                       obs=priors[2].sim)


def spire_model_CIGALE_schect(priors, sed_prior, params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # redshift-sfr relation parameters
    z_star = numpyro.sample('m', dist.TruncatedNormal(0.01,params['z_star_mu'], params['z_star_sig']))
    sfr_star = numpyro.sample('c', dist.TruncatedNormal(0.01,params['sfr_star_mu'], params['sfr_star_sig']))
    alpha= params['alpha']

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(0.01, sed_prior.params_mu[:, 1], sed_prior.params_sig[:, 1]))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        # use handlers.reparam as sampling from standard normal and transforming by redshift relation
        # with numpyro.handlers.reparam(config={"params": TransformReparam()}):
        # sfr = numpyro.sample('sfr', dist.TransformedDistribution(dist.Normal(0.0, 1.0),
        # dist.transforms.AffineTransform(redshift * m + c,
        # jnp.full(
        # priors[0].nsrc,
        # sfr_sig))))
        sfr = numpyro.sample('sfr', dist.Normal((sfr_star*jnp.exp(-1.0*redshift/z_star)*(redshift/z_star)**alpha)-2.0, jnp.full(priors[0].nsrc, sfr_sig)))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_plw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]

    # for each band, condition on data
    with numpyro.plate('psw_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('plw_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[2].snim),
                       obs=priors[2].sim)


def spire_model_CIGALE_kasia_schect(priors, sed_prior, params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # redshift-sfr relation parameters
    z_star = numpyro.sample('m', dist.TruncatedNormal(loc=params['z_star_mu'], scale=params['z_star_sig'],low=0.01,))
    sfr_star = numpyro.sample('c', dist.TruncatedNormal(loc=params['sfr_star_mu'], scale=params['sfr_star_sig'],low=0.01,))
    alpha= params['alpha']

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(loc=sed_prior.params_mu[:, 1],scale= sed_prior.params_sig[:, 1],low=0.01))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal((sfr_star*jnp.exp(-1.0*redshift/z_star)*(redshift/z_star)**alpha)-2.0, jnp.full(priors[0].nsrc, sfr_sig)))

        atten=numpyro.sample('atten',dist.TruncatedNormal(loc=sed_prior.params_mu[:, 2],scale=sed_prior.params_sig[:, 2],low=0.01))

        dust_alpha=numpyro.sample('dust_alpha',dist.TruncatedNormal(loc=sed_prior.params_mu[:, 3], scale=sed_prior.params_sig[:, 3],low=0.01))

        tau_main=numpyro.sample('tau_main',dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, -3][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, -2][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_plw = sp_matmul(pointing_matrices[2], src_f[:, -1][:, None], priors[2].snpix).reshape(-1) + bkg[2]

    # for each band, condition on data
    with numpyro.plate('psw_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('plw_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[2].snim),
                       obs=priors[2].sim)

def spire_model_CIGALE_kasia_schect_bands(priors, sed_prior, params,flux,flux_error):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """


    # redshift-sfr relation parameters
    z_star = numpyro.sample('m', dist.TruncatedNormal(loc=params['z_star_mu'], scale=params['z_star_sig'], low=0.01))
    sfr_star = numpyro.sample('c',
                              dist.TruncatedNormal(loc=params['sfr_star_mu'], scale=params['sfr_star_sig'], low=0.01))
    alpha = params['alpha']

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    plate_sources = numpyro.plate('nsrc', priors[0].nsrc)

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with plate_sources:
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(loc=sed_prior.params_mu[:, 1], scale=sed_prior.params_sig[:, 1],
                                                       low=0.01))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal(
                (sfr_star * jnp.exp(-1.0 * redshift / z_star) * (redshift / z_star) ** alpha) - 2.0,
                jnp.full(priors[0].nsrc, sfr_sig)))

        atten = numpyro.sample('atten',
                               dist.TruncatedNormal(loc=sed_prior.params_mu[:, 2], scale=sed_prior.params_sig[:, 2],
                                                    low=0.0))

        dust_alpha = numpyro.sample('dust_alpha', dist.TruncatedNormal(loc=sed_prior.params_mu[:, 3],
                                                                       scale=sed_prior.params_sig[:, 3], low=0.0))

        tau_main = numpyro.sample('tau_main', dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    with plate_sources:
        # for each band, condition on data
        numpyro.sample("obs_250", dist.Normal(src_f[:, -3], flux_error[0,:]),
                       obs=flux[0,:])
        numpyro.sample("obs_350", dist.Normal(src_f[:, -2], flux_error[1, :]),
                        obs=flux[1, :])
        numpyro.sample("obs_500", dist.Normal(src_f[:, -1], flux_error[2, :]),
                        obs=flux[2, :])



def spire_model_CIGALE_kasia_schect_irac(priors, sed_prior, params,irac_cut,irac_flux,irac_sigma):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # redshift-sfr relation parameters
    z_star = numpyro.sample('m', dist.TruncatedNormal(0.01,params['z_star_mu'], params['z_star_sig']))
    sfr_star = numpyro.sample('c', dist.TruncatedNormal(0.01,params['sfr_star_mu'], params['sfr_star_sig']))
    alpha= params['alpha']

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(0.01, sed_prior.params_mu[:, 1], sed_prior.params_sig[:, 1]))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal((sfr_star*jnp.exp(-1.0*redshift/z_star)*(redshift/z_star)**alpha)-2.0, jnp.full(priors[0].nsrc, sfr_sig)))

        atten=numpyro.sample('atten',dist.TruncatedNormal(0.0, sed_prior.params_mu[:, 2], sed_prior.params_sig[:, 2]))

        dust_alpha=numpyro.sample('dust_alpha',dist.TruncatedNormal(0.0, sed_prior.params_mu[:, 3], sed_prior.params_sig[:, 3]))

        tau_main=numpyro.sample('tau_main',dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, -3][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, -2][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_plw = sp_matmul(pointing_matrices[2], src_f[:, -1][:, None], priors[2].snpix).reshape(-1) + bkg[2]

    with numpyro.plate('nsrc', priors[0].nsrc):
        numpyro.sample('obs_irac',dist.TruncatedNormal(irac_cut,src_f[:,0],irac_sigma),obs=irac_flux)
    # for each band, condition on data
    with numpyro.plate('psw_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('plw_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[2].snim),
                       obs=priors[2].sim)


def SFRD(z,z1,z2,a1,a2,sig1,sig2):
    return a1*jnp.exp(-0.5*jnp.power((z-z1)/sig1,2))+a2*jnp.exp(-0.5*jnp.power((z-z2)/sig2,2))

def spire_model_CIGALE_kasia_SFRD(priors, sed_prior, params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # redshift-sfr relation parameters
    z_1 = numpyro.sample('z1', dist.TruncatedNormal(0.01,params['z_1_mu'], params['z_1_sig']))
    z_2 = numpyro.sample('z2', dist.TruncatedNormal(0.01,params['z_2_mu'], params['z_2_sig']))
    sig_1 = numpyro.sample('sig1',dist.TruncatedNormal(0.01,params['sig_1_mu'], params['sig_1_sig']))
    sig_2 = numpyro.sample('sig2',dist.TruncatedNormal(0.01,params['sig_2_mu'], params['sig_2_sig']))
    a1 = numpyro.sample('a1', dist.TruncatedNormal(0.01,params['a_1_mu'], params['a_1_sig']))
    a2 = numpyro.sample('a2', dist.TruncatedNormal(0.01,params['a_2_mu'], params['a_2_sig']))

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(0.01, sed_prior.params_mu[:, 1], sed_prior.params_sig[:, 1]))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal(jnp.log10(SFRD(redshift,z_1,z_2,a1,a2,sig_1,sig_2)), jnp.full(priors[0].nsrc, sfr_sig)))

        atten=numpyro.sample('atten',dist.TruncatedNormal(0.0, sed_prior.params_mu[:, 2], sed_prior.params_sig[:, 2]))

        dust_alpha=numpyro.sample('dust_alpha',dist.TruncatedNormal(0.0, sed_prior.params_mu[:, 3], sed_prior.params_sig[:, 3]))

        tau_main=numpyro.sample('tau_main',dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, -3][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, -2][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_plw = sp_matmul(pointing_matrices[2], src_f[:, -1][:, None], priors[2].snpix).reshape(-1) + bkg[2]

    # for each band, condition on data
    with numpyro.plate('psw_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('plw_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[2].snim),
                       obs=priors[2].sim)


def spire_model_CIGALE_mainseq(priors, sed_prior, params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # main sequence relation parameters
    alpha = numpyro.sample('alpha',dist.Normal(params['alpha_mu'],params['alpha_sig']))
    beta = numpyro.sample('beta', dist.Normal(params['beta_mu'], params['beta_sig']))

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(0.01, sed_prior.params_mu[:, 1], sed_prior.params_sig[:, 1]))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        m_star=numpyro.sample('mstar',dist.Normal(sed_prior.params_mu[:, 5],sed_prior.params_sig[:, 2]))

        sfr = numpyro.sample('sfr', dist.Normal(alpha*m_star+beta, jnp.full(priors[0].nsrc, sfr_sig)))

        atten=numpyro.sample('atten',dist.TruncatedNormal(0.0, sed_prior.params_mu[:, 2], sed_prior.params_sig[:, 2]))

        dust_alpha=numpyro.sample('dust_alpha',dist.TruncatedNormal(0.0, sed_prior.params_mu[:, 3], sed_prior.params_sig[:, 3]))

        tau_main=numpyro.sample('tau_main',dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:],m_star[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, -3][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, -2][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_plw = sp_matmul(pointing_matrices[2], src_f[:, -1][:, None], priors[2].snpix).reshape(-1) + bkg[2]

    # for each band, condition on data
    with numpyro.plate('psw_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('plw_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[2].snim),
                       obs=priors[2].sim)

def GEP_CIGALE_schect(priors, sed_prior, params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

        # redshift-sfr relation parameters
    z_star = numpyro.sample('m', dist.TruncatedNormal(loc=params['z_star_mu'], scale=params['z_star_sig'], low=0.01))
    sfr_star = numpyro.sample('c',
                              dist.TruncatedNormal(loc=params['sfr_star_mu'], scale=params['sfr_star_sig'], low=0.01))
    alpha = params['alpha']

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(loc=sed_prior.params_mu[:, 1], scale=sed_prior.params_sig[:, 1],
                                                       low=0.01))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal(
            (sfr_star * jnp.exp(-1.0 * redshift / z_star) * (redshift / z_star) ** alpha) - 2.0,
            jnp.full(priors[0].nsrc, sfr_sig)))

        atten = numpyro.sample('atten',
                               dist.TruncatedNormal(loc=sed_prior.params_mu[:, 2], scale=sed_prior.params_sig[:, 2],
                                                    low=0.0))

        dust_alpha = numpyro.sample('dust_alpha', dist.TruncatedNormal(loc=sed_prior.params_mu[:, 3],
                                                                       scale=sed_prior.params_sig[:, 3], low=0.0))

        tau_main=numpyro.sample('tau_main',dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_19 = sp_matmul(pointing_matrices[0], src_f[:, 18][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_20 = sp_matmul(pointing_matrices[1], src_f[:, 19][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_21 = sp_matmul(pointing_matrices[2], src_f[:, 20][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_22 = sp_matmul(pointing_matrices[3], src_f[:, 21][:, None], priors[3].snpix).reshape(-1) + bkg[3]
    db_hat_23 = sp_matmul(pointing_matrices[4], src_f[:, 22][:, None], priors[4].snpix).reshape(-1) + bkg[4]

    # for each band, condition on data
    with numpyro.plate('gep19_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_gep19", dist.Normal(db_hat_19, priors[0].snim),
                       obs=priors[0].sim)
    with numpyro.plate('gep20_pixels', priors[1].snim.size):  # as ind_pmw:
        numpyro.sample("obs_gep20", dist.Normal(db_hat_20, priors[1].snim),
                       obs=priors[1].sim)
    with numpyro.plate('gep21_pixels', priors[2].snim.size):  # as ind_plw:
        numpyro.sample("obs_gep21", dist.Normal(db_hat_21, priors[2].snim),
                       obs=priors[2].sim)
    with numpyro.plate('gep22_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_gep22", dist.Normal(db_hat_22, priors[3].snim),
                       obs=priors[3].sim)
    with numpyro.plate('gep23_pixels', priors[4].snim.size):  # as ind_plw:
        numpyro.sample("obs_gep23", dist.Normal(db_hat_23, priors[4].snim),
                       obs=priors[4].sim)

def GEP_CIGALE_schect_19(priors, sed_prior, params):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    # get pointing matices in useable format
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]

    # get background priors for maps
    bkg_mu = np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    # background priors
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

    # redshift-sfr relation parameters
    z_star = numpyro.sample('m', dist.TruncatedNormal(loc=params['z_star_mu'], scale=params['z_star_sig'],low=0.01))
    sfr_star = numpyro.sample('c', dist.TruncatedNormal(loc=params['sfr_star_mu'],scale=params['sfr_star_sig'],low=0.01))
    alpha= params['alpha']

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with numpyro.plate('nsrc', priors[0].nsrc):
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(loc= sed_prior.params_mu[:, 1],scale=sed_prior.params_sig[:, 1],low=0.01))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal((sfr_star*jnp.exp(-1.0*redshift/z_star)*(redshift/z_star)**alpha)-2.0, jnp.full(priors[0].nsrc, sfr_sig)))

        atten=numpyro.sample('atten',dist.TruncatedNormal(loc=sed_prior.params_mu[:, 2], scale=sed_prior.params_sig[:, 2],low=0.0))

        dust_alpha=numpyro.sample('dust_alpha',dist.TruncatedNormal(loc=sed_prior.params_mu[:, 3], scale=sed_prior.params_sig[:, 3],low=0.0))

        tau_main=numpyro.sample('tau_main',dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    # create model map by multiplying fluxes by pointing matrix and adding background
    db_hat_19 = sp_matmul(pointing_matrices[0], src_f[:, 18][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    # db_hat_20 = sp_matmul(pointing_matrices[1], src_f[:, 19][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    # db_hat_21 = sp_matmul(pointing_matrices[2], src_f[:, 20][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    # db_hat_22 = sp_matmul(pointing_matrices[3], src_f[:, 21][:, None], priors[3].snpix).reshape(-1) + bkg[3]
    # db_hat_23 = sp_matmul(pointing_matrices[4], src_f[:, 22][:, None], priors[4].snpix).reshape(-1) + bkg[4]

    # for each band, condition on data
    with numpyro.plate('gep19_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_gep19", dist.Normal(db_hat_19, priors[0].snim),
                       obs=priors[0].sim)
    # with numpyro.plate('gep20_pixels', priors[1].snim.size):  # as ind_pmw:
    #     numpyro.sample("obs_gep20", dist.Normal(db_hat_20, priors[1].snim),
    #                    obs=priors[1].sim)
    # with numpyro.plate('gep21_pixels', priors[2].snim.size):  # as ind_plw:
    #     numpyro.sample("obs_gep21", dist.Normal(db_hat_21, priors[2].snim),
    #                    obs=priors[2].sim)
    # with numpyro.plate('gep22_pixels', priors[3].snim.size):  # as ind_plw:
    #     numpyro.sample("obs_gep22", dist.Normal(db_hat_22, priors[3].snim),
    #                    obs=priors[3].sim)
    # with numpyro.plate('gep23_pixels', priors[4].snim.size):  # as ind_plw:
    #     numpyro.sample("obs_gep23", dist.Normal(db_hat_23, priors[4].snim),
    #                    obs=priors[4].sim)

def GEP_CIGALE_schect_bands(priors, sed_prior, params,flux,flux_error):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """




        # redshift-sfr relation parameters
    z_star = numpyro.sample('m', dist.TruncatedNormal(loc=params['z_star_mu'], scale=params['z_star_sig'], low=0.01))
    sfr_star = numpyro.sample('c',
                              dist.TruncatedNormal(loc=params['sfr_star_mu'], scale=params['sfr_star_sig'], low=0.01))
    alpha = params['alpha']

    # sfr dispersion parameter
    sfr_sig = numpyro.sample('sfr_sig', dist.HalfNormal(params['sfr_disp']))

    plate_sources=numpyro.plate('nsrc',priors[0].nsrc)

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with plate_sources:
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(loc=sed_prior.params_mu[:, 1], scale=sed_prior.params_sig[:, 1],
                                                       low=0.01))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal(
            (sfr_star * jnp.exp(-1.0 * redshift / z_star) * (redshift / z_star) ** alpha) - 2.0,
            jnp.full(priors[0].nsrc, sfr_sig)))

        atten = numpyro.sample('atten',
                               dist.TruncatedNormal(loc=sed_prior.params_mu[:, 2], scale=sed_prior.params_sig[:, 2],
                                                    low=0.0))

        dust_alpha = numpyro.sample('dust_alpha', dist.TruncatedNormal(loc=sed_prior.params_mu[:, 3],
                                                                       scale=sed_prior.params_sig[:, 3], low=0.0))

        tau_main=numpyro.sample('tau_main',dist.Normal(sed_prior.params_mu[:, 4], sed_prior.params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(sed_prior.emulator['net_apply'](sed_prior.emulator['params'], params))

    with plate_sources:

    # for each band, condition on data
        numpyro.sample("obs_gep19", dist.Normal(src_f[:, 18], flux_error[0,:]),
                       obs=flux[0,:])
        numpyro.sample("obs_gep20", dist.Normal(src_f[:, 19], flux_error[1,:]),
                       obs=flux[1,:])
        numpyro.sample("obs_gep21", dist.Normal(src_f[:, 20], flux_error[2,:]),
                       obs=flux[2,:])
        numpyro.sample("obs_gep22", dist.Normal(src_f[:, 21], flux_error[3,:]),
                       obs=flux[3,:])
        numpyro.sample("obs_gep23", dist.Normal(src_f[:, 22], flux_error[4,:]),
                       obs=flux[4,:])


def GEP_CIGALE_bands(emulator,params_mu,params_sig,flux,flux_error):
    """
    numpyro model for SPIRE maps using cigale emulator
    :param priors: list of xid+ SPIRE prior objects
    :type priors: list
    :param sed_prior: xid+ SED prior class
    :type sed_prior:
    :return:
    :rtype:
    """

    plate_sources=numpyro.plate('nsrc',params_mu.shape[0])

    # sample parameters for each source (treat as conditionaly independent hence plate)
    with plate_sources:
        # use truncated normal for redshift, with mean and sigma from prior
        redshift = numpyro.sample('redshift',
                                  dist.TruncatedNormal(loc=params_mu[:, 1], scale=params_sig[:, 1],
                                                       low=0.01))
        # use beta distribution for AGN as a fraction
        agn = numpyro.sample('agn', dist.Beta(1.0, 3.0))

        sfr = numpyro.sample('sfr', dist.Normal(loc=params_mu[:,0],scale=params_sig[:,0]))

        atten = numpyro.sample('atten',
                               dist.TruncatedNormal(loc=params_mu[:, 2], scale=params_sig[:, 2],
                                                    low=0.0))

        dust_alpha = numpyro.sample('dust_alpha', dist.TruncatedNormal(loc=params_mu[:, 3],
                                                                       scale=params_sig[:, 3], low=0.0))

        tau_main=numpyro.sample('tau_main',dist.Normal(params_mu[:, 4], params_sig[:, 4]))

    # stack params and make vector ready to be used by emualator
    params = jnp.vstack((sfr[None, :], agn[None, :], redshift[None, :], atten[None,:],dust_alpha[None,:],tau_main[None,:])).T
    # Use emulator to get fluxes. As emulator provides log flux, convert.
    src_f = jnp.exp(emulator['net_apply'](emulator['params'], params))

    with plate_sources:

    # for each band, condition on data
        numpyro.sample("obs_gep19", dist.Normal(src_f[:, 18], flux_error[0,:]),
                       obs=flux[0,:])
        numpyro.sample("obs_gep20", dist.Normal(src_f[:, 19], flux_error[1,:]),
                       obs=flux[1,:])
        numpyro.sample("obs_gep21", dist.Normal(src_f[:, 20], flux_error[2,:]),
                       obs=flux[2,:])
        numpyro.sample("obs_gep22", dist.Normal(src_f[:, 21], flux_error[3,:]),
                       obs=flux[3,:])
        numpyro.sample("obs_gep23", dist.Normal(src_f[:, 22], flux_error[4,:]),
                       obs=flux[4,:])