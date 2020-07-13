import torch
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO



def spire_model(priors):

    if len(priors) != 3:
        raise ValueError
    band_plate = pyro.plate('bands', len(priors), dim=-2)
    src_plate = pyro.plate('nsrc', priors[0].nsrc, dim=-1)
    psw_plate = pyro.plate('psw_pixels', priors[0].sim.size, dim=-3)
    pmw_plate = pyro.plate('pmw_pixels', priors[1].sim.size, dim=-3)
    plw_plate = pyro.plate('plw_pixels', priors[2].sim.size, dim=-3)
    pointing_matrices = [torch.sparse.FloatTensor(
            torch.LongTensor([p.amat_row, p.amat_col]),
            torch.Tensor(p.amat_data), torch.Size([p.snpix, p.nsrc])) for p in priors]

    bkg_prior = torch.tensor([p.bkg[0] for p in priors])
    bkg_prior_sig = torch.tensor([p.bkg[1] for p in priors])
    nsrc = priors[0].nsrc

    f_low_lim = torch.tensor([p.prior_flux_lower for p in priors], dtype=torch.float)
    f_up_lim = torch.tensor([p.prior_flux_upper for p in priors], dtype=torch.float)

    with band_plate as ind_band:
        sigma_conf = pyro.sample('sigma_conf', dist.Exponential(1).expand([1]).to_event(1)).squeeze(-1)
        bkg = pyro.sample('bkg', dist.Normal(-5, 0.5).expand([1]).to_event(1)).squeeze(-1)
        with src_plate as ind_src:
            src_f = pyro.sample('src_f', dist.Uniform(0, 1).expand([1]).to_event(1)).squeeze(-1)
    f_vec = (f_up_lim - f_low_lim) * src_f + f_low_lim
    db_hat_psw = torch.sparse.mm(pointing_matrices[0], f_vec[0, ...].unsqueeze(-1)) + bkg[0]
    db_hat_pmw = torch.sparse.mm(pointing_matrices[1].to_dense(), f_vec[1, ...].unsqueeze(-1)) + bkg[1]
    db_hat_plw = torch.sparse.mm(pointing_matrices[2].to_dense(), f_vec[2, ...].unsqueeze(-1)) + bkg[2]
    sigma_tot_psw = torch.sqrt(torch.pow(torch.tensor(priors[0].snim), 2) + torch.pow(sigma_conf[0], 2))
    sigma_tot_pmw = torch.sqrt(torch.pow(torch.tensor(priors[1].snim), 2) + torch.pow(sigma_conf[1], 2))
    sigma_tot_plw = torch.sqrt(torch.pow(torch.tensor(priors[2].snim), 2) + torch.pow(sigma_conf[2], 2))
    with psw_plate:
        psw_map = pyro.sample("obs_psw", dist.Normal(db_hat_psw.squeeze(), sigma_tot_psw),
                              obs=torch.tensor(priors[0].sim))
    with pmw_plate:
        pmw_map = pyro.sample("obs_pmw", dist.Normal(db_hat_pmw.squeeze(), sigma_tot_pmw),
                              obs=torch.tensor(priors[1].sim))
    with plw_plate:
        plw_map = pyro.sample("obs_plw", dist.Normal(db_hat_plw.squeeze(), sigma_tot_plw),
                              obs=torch.tensor(priors[2].sim))
    return psw_map, pmw_map, plw_map



def all_bands(priors,lr=0.005,n_steps=1000,n_samples=1000,verbose=True):
    from pyro.infer import Predictive

    pyro.clear_param_store()

    guide = AutoMultivariateNormal(spire_model, init_loc_fn=init_to_mean)

    svi = SVI(spire_model,
              guide,
              optim.Adam({"lr":lr}),
              loss=Trace_ELBO())

    loss_history = []
    for i in range(n_steps):
        loss = svi.step(priors)
        if (i % 100 == 0) and verbose:
            print('ELBO loss: {}'.format(loss))
        loss_history.append(loss)
    print('ELBO loss: {}'.format(loss))
    predictive = Predictive(spire_model, guide=guide, num_samples=n_samples)
    samples = {k: v.squeeze(-1).detach().cpu().numpy() for k, v in predictive(priors).items()
                     if k != "obs"}
    f_low_lim = torch.tensor([p.prior_flux_lower for p in priors], dtype=torch.float)
    f_up_lim = torch.tensor([p.prior_flux_upper for p in priors], dtype=torch.float)
    f_vec_multi = (f_up_lim - f_low_lim) * samples['src_f'][..., :, :] + f_low_lim
    samples['src_f'] = f_vec_multi.squeeze(-3).numpy()
    samples['sigma_conf']=samples['sigma_conf'].squeeze(-1).squeeze(-2)
    samples['bkg'].squeeze(-1).squeeze(-2)

    return {'loss_history':loss_history,'samples':samples}
