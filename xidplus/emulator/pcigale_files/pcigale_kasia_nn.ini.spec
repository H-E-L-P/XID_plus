data_file = string()
parameters_file = string()
sed_modules = cigale_string_list()
analysis_method = string()
cores = integer(min=1)
bands = cigale_string_list()
properties = cigale_string_list()
[sed_modules_params]
  [[sfhdelayed]]
    tau_main = cigale_list()
    age_main = cigale_list(dtype=int, minvalue=0.)
    tau_burst = cigale_list()
    age_burst = cigale_list(dtype=int, minvalue=1.)
    f_burst = cigale_list(minvalue=0., maxvalue=0.9999)
    sfr_A = cigale_list(minvalue=0.)
    normalise = boolean()
  [[bc03]]
    imf = cigale_list(dtype=int, options=0. & 1.)
    metallicity = cigale_list(options=0.0001 & 0.0004 & 0.004 & 0.008 & 0.02 & 0.05)
    separation_age = cigale_list(dtype=int, minvalue=0)
  [[dustatt_2powerlaws]]
    Av_BC = cigale_list(minvalue=0)
    slope_BC = cigale_list()
    BC_to_ISM_factor = cigale_list(minvalue=0., maxvalue=1.)
    slope_ISM = cigale_list()
    filters = string()
  [[lyc_absorption]]
    f_esc = cigale_list(minvalue=0., maxvalue=1.)
    f_dust = cigale_list(minvalue=0., maxvalue=1.)
  [[dale2014]]
    fracAGN = cigale_list(minvalue=0., maxvalue=1.)
    alpha = cigale_list(options=0.0625 & 0.1250 & 0.1875 & 0.2500 & 0.3125 & 0.3750 & 0.4375 & 0.5000 & 0.5625 & 0.6250 & 0.6875 & 0.7500 & 0.8125 & 0.8750 & 0.9375 & 1.0000 & 1.0625 & 1.1250 & 1.1875 & 1.2500 & 1.3125 & 1.3750 & 1.4375 & 1.5000 & 1.5625 & 1.6250 & 1.6875 & 1.7500 & 1.8125 & 1.8750 & 1.9375 & 2.0000 & 2.0625 & 2.1250 & 2.1875 & 2.2500 & 2.3125 & 2.3750 & 2.4375 & 2.5000 & 2.5625 & 2.6250 & 2.6875 & 2.7500 & 2.8125 & 2.8750 & 2.9375 & 3.0000 & 3.0625 & 3.1250 & 3.1875 & 3.2500 & 3.3125 & 3.3750 & 3.4375 & 3.5000 & 3.5625 & 3.6250 & 3.6875 & 3.7500 & 3.8125 & 3.8750 & 3.9375 & 4.0000)
  [[fritz2006]]
    r_ratio = cigale_list(options=10. & 30. & 60. & 100. & 150.)
    tau = cigale_list(options=0.1 & 0.3 & 0.6 & 1.0 & 2.0 & 3.0 & 6.0 & 10.0)
    beta = cigale_list(options=-1.00 & -0.75 & -0.50 & -0.25 & 0.00)
    gamma = cigale_list(options=0.0 & 2.0 & 4.0 & 6.0)
    opening_angle = cigale_list(options=60. & 100. & 140.)
    psy = cigale_list(options=0.001 & 10.100 & 20.100 & 30.100 & 40.100 & 50.100 & 60.100 & 70.100 & 80.100 & 89.990)
    fracAGN = cigale_list(minvalue=0., maxvalue=1.)
  [[redshifting]]
    redshift = cigale_list(minvalue=0.)
[analysis_params]
  variables = cigale_string_list()
  save_sed = boolean()
  blocks = integer(min=1)
