_author__ = 'pdh21'

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

def pacs_model(priors):
    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]
    flux_lower = np.asarray([p.prior_flux_lower for p in priors]).T
    flux_upper = np.asarray([p.prior_flux_upper for p in priors]).T

    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T

    with numpyro.plate('bands', len(priors)):
        sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(1.0, 0.5))
        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

        with numpyro.plate('nsrc', priors[0].nsrc):
            src_f = numpyro.sample('src_f', dist.Uniform(flux_lower, flux_upper))
    db_hat_psw = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]
    db_hat_pmw = sp_matmul(pointing_matrices[1], src_f[:, 1][:, None], priors[1].snpix).reshape(-1) + bkg[1]

    sigma_tot_psw = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))
    sigma_tot_pmw = jnp.sqrt(jnp.power(priors[1].snim, 2) + jnp.power(sigma_conf[1], 2))

    with numpyro.plate('psw_pixels', priors[0].sim.size):  # as ind_psw:
        numpyro.sample("obs_psw", dist.Normal(db_hat_psw, sigma_tot_psw),
                                 obs=priors[0].sim)
    with numpyro.plate('pmw_pixels', priors[1].sim.size):  # as ind_pmw:
        numpyro.sample("obs_pmw", dist.Normal(db_hat_pmw, sigma_tot_pmw),
                                 obs=priors[1].sim)


def all_bands(priors,num_samples=500,num_warmup=500,num_chains=4,chain_method='parallel'):
    numpyro.set_host_device_count(4)
    nuts_kernel = NUTS(pacs_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,num_chains=num_chains,chain_method=chain_method)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, priors)
    return mcmc