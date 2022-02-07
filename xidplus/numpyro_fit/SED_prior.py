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


