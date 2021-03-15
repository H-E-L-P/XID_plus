_author__ = 'pdh21'

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import TransformReparam
import jax.numpy as jnp
from jax import random
import numpy as np
import jax
import pickle
import os
from xidplus.numpyro_fit.misc import sp_matmul, load_emulator

numpyro.set_host_device_count(os.cpu_count())


def spire_model(priors,sed_prior):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 3):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))

    #params = jnp.vstack((sfr[None,:],agn_frac[None,:],redshift[None,:])).T

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands(priors,sed_prior,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,extra_fields=('potential_energy',))
    return mcmc


def spire_model_fix_z(priors,sed_prior,redshift):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 2):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))

    params_tmp = jnp.hstack((params,redshift[:,None]))

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_fix_z(priors,sed_prior,redshift,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_fix_z)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,redshift,extra_fields=('potential_energy',))
    return mcmc

def spire_model_fix_sfr(priors,sed_prior,sfr):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 2):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))

    params_tmp = jnp.hstack((sfr[:,None],params))

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_fix_sfr(priors,sed_prior,sfr,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_fix_sfr)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,sfr,extra_fields=('potential_energy',))
    return mcmc

def spire_model_fix_sfr_redshift(priors,sed_prior,sfr,redshift):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 1):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))

    params_tmp = jnp.hstack((sfr[:,None],params,redshift[:,None]))

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_fix_sfr_redshift(priors,sed_prior,sfr,redshift,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_fix_sfr_redshift)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,sfr,redshift,extra_fields=('potential_energy',))
    return mcmc

def spire_model_fix_params(priors,sed_prior,params):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    #with numpyro.plate('n_param', 1):
     #   with numpyro.plate('nsrc', priors[0].nsrc):
     #       with numpyro.handlers.reparam(config={"params":TransformReparam()}):
     #           params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))



    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_fix_params(priors,sed_prior,params,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_fix_params)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,params,extra_fields=('potential_energy',))
    return mcmc

def spire_model_fix_agn_redshift(priors,sed_prior,agn,redshift):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 1):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))

    params_tmp = jnp.hstack((params, agn[:,None], redshift[:, None]))

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_fix_agn_redshift(priors,sed_prior,agn,redshift,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_fix_agn_redshift)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,agn,redshift,extra_fields=('potential_energy',))
    return mcmc

def spire_model_fix_agn(priors,sed_prior,agn):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 2):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))
    params_tmp = jnp.hstack((params[:,0][:,None], agn[:,None], params[:,1][:,None]))

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_fix_agn(priors,sed_prior,agn,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_fix_agn)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,agn,extra_fields=('potential_energy',))
    return mcmc

def spire_model_fix_sfr_agn(priors,sed_prior,sfr,agn):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 1):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))

    params_tmp = jnp.hstack((sfr[:,None], agn[:,None], params))

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_fix_sfr_agn(priors,sed_prior,sfr,agn,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_fix_sfr_agn)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,sfr,agn,extra_fields=('potential_energy',))
    return mcmc

def spire_model_uniAGN(priors,sed_prior):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))



    with numpyro.plate('nsrc', priors[0].nsrc):
        with numpyro.plate('n_param', 2):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu.T,sed_prior.params_sig.T)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        agn_frac=numpyro.sample('agn_frac',dist.Uniform(0.0,1.0))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))
    params_tmp = jnp.hstack((params[0, :][:, None], agn_frac[:, None], params[1, :][:, None]))
    #params = jnp.vstack((sfr[None,:],agn_frac[None,:],redshift[None,:])).T

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 2][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 3][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))

    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_uniAGN(priors,sed_prior,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    #reparam_model=reparam(spire_model, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(spire_model_uniAGN)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,extra_fields=('potential_energy',))
    return mcmc

def irac_spire_model_fix_agn(priors,sed_prior,agn,f_irac1,f_irac1_sig):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]


    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    #lir_mu=numpyro.sample('lir_mu',dist.Normal(10,1))
    #lir_sig=numpyro.sample('lir_sig',dist.HalfCauchy(1,0.5))

    #params_mu=sed_prior.params_mu
    #params_mu[:,0]=lir_mu
    #params_sig=sed_prior.params_sig
    #params_sig[:,0]=lir_sig
    with numpyro.plate('bands', len(priors)):

        #sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1,0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu,bkg_sig))


    with numpyro.plate('n_param', 2):
        with numpyro.plate('nsrc', priors[0].nsrc):
            with numpyro.handlers.reparam(config={"params":TransformReparam()}):
                params = numpyro.sample('params', dist.TransformedDistribution(dist.Normal(0.0,1.0),dist.transforms.AffineTransform(sed_prior.params_mu,sed_prior.params_sig)))
        #sfr=numpyro.sample('sfr',dist.Normal(sed_prior.params_mu[:,0],sed_prior.params_sig[:,0]))
        #agn_frac=numpyro.sample('agn_frac',dist.Normal(sed_prior.params_mu[:,1],sed_prior.params_sig[:,1]))
        #redshift=numpyro.sample('redshift',dist.Normal(sed_prior.params_mu[:,2],sed_prior.params_sig[:,2]))
    params_tmp = jnp.hstack((params[:,0][:,None], agn[:,None], params[:,1][:,None]))

    src_f=jnp.power(10.0,sed_prior.emulator['net_apply'](sed_prior.emulator['params'],params_tmp))

    #src_f = numpyro.sample('src_f', dist.(flux_lower, flux_upper))
    db_hat_mips24 = sp_matmul(pointing_matrices[0], src_f[:, 1][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_psw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]
    db_hat_pmw = sp_matmul(pointing_matrices[2], src_f[:, 3][:, None], priors[2].snpix).reshape(-1) + bkg[2]
    db_hat_plw = sp_matmul(pointing_matrices[3], src_f[:, 4][:, None], priors[3].snpix).reshape(-1) + bkg[3]

    #sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    #sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))
    #sigma_tot_plw = jnp.sqrt(jnp.power(priors[2].snim, 2) + jnp.power(sigma_conf[2], 2))
    with numpyro.plate('f_irac1_sources',f_irac1_sig.size):
        numpyro.sample("obs_f_irac1",dist.Normal(src_f[:,0],f_irac1_sig),
                       obs=f_irac1)
    with numpyro.plate('mips24_pixels', priors[0].snim.size):  # as ind_psw:
        numpyro.sample("obs_mips24", dist.Normal(db_hat_mips24,priors[0].snim ),
                       obs=priors[0].sim)
    with numpyro.plate('psw_pixels', priors[1].snim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw,priors[1].snim ),
                                 obs=priors[1].sim)
    with numpyro.plate('pmw_pixels', priors[2].snim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, priors[2].snim),
                                 obs=priors[2].sim)
    with numpyro.plate('plw_pixels', priors[3].snim.size):  # as ind_plw:
        numpyro.sample("obs_plw", dist.Normal(db_hat_plw, priors[3].snim),
                                 obs=priors[3].sim)
    return [db_hat_mips24,db_hat_psw,db_hat_pmw,db_hat_plw]

def all_bands_irac_fix_agn(priors,sed_prior,agn,f_irac1,f_irac1_sig,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.handlers import reparam
    numpyro.set_host_device_count(os.cpu_count())
    reparam_model=reparam(irac_spire_model_fix_agn, config={'params': LocScaleReparam(0)})
    nuts_kernel = NUTS(reparam_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors,sed_prior,agn,f_irac1,f_irac1_sig,extra_fields=('potential_energy',))
    return mcmc