__author__ = 'pdh21'

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from jax import random
import numpy as np
import jax
import os
from xidplus.numpyro_fit.misc import sp_matmul

numpyro.set_host_device_count(os.cpu_count())

def mips_model(priors):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]
    flux_lower = np.asarray([p.prior_flux_lower for p in priors]).T
    flux_upper = np.asarray([p.prior_flux_upper for p in priors]).T

    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    with numpyro.plate('bands', len(priors)):
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))
        with numpyro.plate('nsrc', priors[0].nsrc):
            src_f = numpyro.sample('src_f', dist.Uniform(flux_lower, flux_upper))
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]

    sigma_tot_psw = priors[0].snim

    with numpyro.plate('psw_pixels', priors[0].sim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, sigma_tot_psw),
                                 obs=priors[0].sim)

def MIPS_24(prior,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    numpyro.set_host_device_count(4)
    nuts_kernel = NUTS(mips_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, [prior])
    return mcmc