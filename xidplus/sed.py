import pandas as pd
import numpy as np


class posterior_sed():
    def __init__(self, fit, priors,sed_prior_model,scale=True):
        import xidplus.stan_fit.stan_utility as stan_utility
        stan_utility.check_treedepth(fit)
        stan_utility.check_energy(fit)
        stan_utility.check_div(fit)
        self.nsrc = priors[0].nsrc
        self.samples = fit.extract()
        self.samples['src_f']=np.swapaxes(self.samples['src_f'], 1, 2)

        nondiv_params, div_params = stan_utility.partition_div(fit)

        self.nondiv_params=nondiv_params
        self.div_params=div_params
        self.param_names = fit.model_pars
        self.summary=fit.summary()
        self.ID = priors[0].ID
        self.Rhat = {'src_f': fit.summary('src_f')['summary'][:, -1].reshape(priors[0].nsrc, len(priors)),
                     'sigma_conf': fit.summary('sigma_conf')['summary'][:, -1],
                     'bkg': fit.summary('bkg')['summary'][:, -1],'Nbb':fit.summary('Nbb')['summary'][:, -1],
                      'z':fit.summary('z')['summary'][:, -1]}

        self.n_eff = {'src_f': fit.summary('src_f')['summary'][:, -2].reshape(priors[0].nsrc, len(priors)),
                      'sigma_conf': fit.summary('sigma_conf')['summary'][:, -2],
                      'bkg': fit.summary('bkg')['summary'][:, -2],'Nbb':fit.summary('Nbb')['summary'][:, -2],
                      'z':fit.summary('z')['summary'][:, -2]}



def berta_templates(SPIRE=True, PACS=True, MIPS=True):
    import os
    import numpy as np
    from astropy.io import ascii
    from scipy.interpolate import interp1d
    import xidplus

    temps = os.listdir(xidplus.__path__[0]+'/../test_files/templates_berta_norm_LIR/')

    #Generate Redshift Grid and convert to denominator for flux conversion(e.g. $4 \pi D_l ^ 2)$
    red = np.arange(0, 8, 0.01)
    red[0] = 0.000001
    from astropy.cosmology import Planck13
    import astropy.units as u
    div = (4.0 * np.pi * np.square(Planck13.luminosity_distance(red).cgs))
    div = div.value

    #Get appropriate filters
    from xidplus import filters
    filter = filters.FilterFile(file=xidplus.__path__[0] + '/../test_files/filters.res')

    SPIRE_250 = filter.filters[215]
    SPIRE_350 = filter.filters[216]
    SPIRE_500 = filter.filters[217]
    MIPS_24 = filter.filters[201]
    PACS_100 = filter.filters[250]
    PACS_160 = filter.filters[251]

    bands = []
    eff_lam=[]
    if MIPS is True:
        bands.extend([MIPS_24])
        eff_lam.extend([24.0])

    if PACS is True:
        bands.extend([PACS_100,PACS_160])
        eff_lam.extend([100.0,160.0])

    if SPIRE is True:
        bands.extend([SPIRE_250,SPIRE_350,SPIRE_500])
        eff_lam.extend([250.0,350.0,500.0])

    print(eff_lam)
    import pandas as pd
    template = ascii.read(xidplus.__path__[0]+'/../test_files/templates_berta_norm_LIR/' + temps[0])
    df = pd.DataFrame(template['col1'].data / 1E4, columns=['wave'])
    SEDs = np.empty((len(temps), len(bands), red.size))
    for i in range(0, len(temps)):
        template = ascii.read(xidplus.__path__[0]+'/../test_files/templates_berta_norm_LIR/' + temps[i])
        df[temps[i]] = 1E30 * 3.826E33 * template['col2'] * ((template['col1'] / 1E4) ** 2) / 3E14

        flux = template['col2'] * ((template['col1'] / 1E4) ** 2) / 3E14
        wave = template['col1'] / 1E4

        for z in range(0, red.size):
            sed = interp1d((red[z] + 1.0) * wave, flux)
            for b in range(0, len(bands)):
                SEDs[i, b, z] = 1E30 * 3.826E33 * (1.0 + red[z]) * filters.fnu_filt(sed(bands[b].wavelength / 1E4),
                                                                                    3E8 / (bands[b].wavelength / 1E10),
                                                                                    bands[b].transmission,
                                                                                    3E8 / (eff_lam[b] * 1E-6),
                                                                                    sed(eff_lam[b])) / div[z]
    return SEDs, df

def mrr_templates(SPIRE=True, PACS=True, MIPS=True):
    import os
    import numpy as np
    from astropy.io import ascii
    from astropy.table import Table

    from scipy.interpolate import interp1d
    import xidplus

    temps = xidplus.__path__[0] + '/../test_files/MRR/'

    # Generate Redshift Grid and convert to denominator for flux conversion(e.g. $4 \pi D_l ^ 2)$
    red = np.arange(0, 8, 0.1)
    red[0] = 0.000001
    from astropy.cosmology import Planck13
    import astropy.units as u
    div = (4.0 * np.pi * np.square(Planck13.luminosity_distance(red).cgs))
    div = div.value

    # Get appropriate filters
    from xidplus import filters
    filter = filters.FilterFile(file=xidplus.__path__[0] + '/../test_files/filters.res')

    SPIRE_250 = filter.filters[215]
    SPIRE_350 = filter.filters[216]
    SPIRE_500 = filter.filters[217]
    MIPS_24 = filter.filters[201]
    PACS_100 = filter.filters[250]
    PACS_160 = filter.filters[251]

    bands = []
    eff_lam = []
    if MIPS is True:
        bands.extend([MIPS_24])
        eff_lam.extend([24.0])

    if PACS is True:
        bands.extend([PACS_100, PACS_160])
        eff_lam.extend([100.0, 160.0])

    if SPIRE is True:
        bands.extend([SPIRE_250, SPIRE_350, SPIRE_500])
        eff_lam.extend([250.0, 350.0, 500.0])

    cirrus = Table.read(temps+'cirrus.dat', format='ascii')
    dusttor = Table.read(temps+'dusttor.dat', format='ascii')
    M82 = Table.read(temps+'M82.dat', format='ascii')
    A220 = Table.read(temps+'A220.dat', format='ascii')
    dusttor['col2'] = np.log(dusttor['col2'])
    cirrus.add_row([0.1, -15])
    M82.add_row([0.1, -10])

    import pandas as pd
    df_comb = pd.DataFrame(np.power(10.0, cirrus['col1'].data), columns=['wave'])
    MRR_temps = [cirrus, A220, M82, dusttor]
    SEDs_comb = np.empty((len(MRR_temps), len(bands), red.size))

    for i in range(0, len(MRR_temps)):

        flux = np.power(10.0, MRR_temps[i]['col2']) / (3.0E14 / np.power(10.0, MRR_temps[i]['col1']))

        wave = np.power(10.0, MRR_temps[i]['col1'])
        ind = (wave > 8) & (wave < 1E3)
        flux = 1E-4 * flux / np.trapz(flux[ind], x=3E14 / wave[ind])
        print(np.trapz(flux[ind], x=3E14 / wave[ind]))
        sed = interp1d(wave, 1E30 * 3.826E33 * flux)
        df_comb[str(i)] = sed(df_comb['wave'])

        for z in range(0, red.size):
            sed = interp1d((red[z] + 1.0) * wave, flux)
            for b in range(0, len(bands)):

                SEDs_comb[i, b, z] = 1E30 * 3.826E33 * (1.0 + red[z]) * filters.fnu_filt(
                    sed(bands[b].wavelength / 1E4), 3E8 / (bands[b].wavelength / 1E10), bands[b].transmission,
                    3E8 / (eff_lam[b] * 1E-6), sed(eff_lam[b])) / div[z]

    return SEDs_comb, df_comb
