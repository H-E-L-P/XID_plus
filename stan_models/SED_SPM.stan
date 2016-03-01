//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  vector[nsrc] f_low_lim_psw;//upper limit of flux 
  vector[nsrc] f_up_lim_psw;//upper limit of flux 
  //----PMW----
  vector[nsrc] f_low_lim_pmw;//upper limit of flux (in log10)
  vector[nsrc] f_up_lim_pmw;//upper limit of flux (in log10)
  //----PLW----
  vector[nsrc] f_low_lim_plw;//upper limit of flux 
  vector[nsrc] f_up_lim_plw;//upper limit of flux
  //-----SED-prior--------
  int<lower=0> nz;// number of points in zpdf
  vector[nz] logpz[nsrc];///logpdf for each source
  int<lower=0> nlam;//number of points in SED
  int<lower=0> nSED;//number of SED temps
  real SEDs[nlam,nSED]; // SEDs
  vector[3] sigma;
 
}
transformed data {
  vector[3] tmp_sed[nz,nSED];
  for (s in 1:nSED){
      for (z in 1:nz) {
	tmp_sed[z,s,1] <- SEDs[z+2*nz,s]/SEDs[z,s];
	tmp_sed[z,s,2] <- SEDs[z+2*nz,s]/SEDs[z+nz,s];
	tmp_sed[z,s,3] <- SEDs[z+nz,s]/SEDs[z,s];
      }}
}

parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f_psw;//source vector
  real bkg_psw;//background
  vector<lower=0.0,upper=1.0>[nsrc] src_f_pmw;//source vector
  real bkg_pmw;//background
  vector<lower=0.0,upper=1.0>[nsrc] src_f_plw;//source vector
  real bkg_plw;//background
  real<lower=0.0,upper=8> sigma_conf_psw;
  real<lower=0.0,upper=8> sigma_conf_pmw;
  real<lower=0.0,upper=8> sigma_conf_plw;
  

}

model {

  row_vector[3] f_vec[nsrc];//matrix of source fluxes

  vector[nz] ps;//log probs
  vector[3] tmp_fvec;
 

  
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec[n,1] <- f_low_lim_psw[n]+(f_up_lim_psw[n]-f_low_lim_psw[n])*src_f_psw[n];
    f_vec[n,2] <- f_low_lim_pmw[n]+(f_up_lim_pmw[n]-f_low_lim_pmw[n])*src_f_pmw[n];
    f_vec[n,3] <- f_low_lim_plw[n]+(f_up_lim_plw[n]-f_low_lim_plw[n])*src_f_plw[n];
    tmp_fvec[1] <- f_vec[n,3]/f_vec[n,1];
    tmp_fvec[2] <- f_vec[n,3]/f_vec[n,2];
    tmp_fvec[3] <- f_vec[n,2]/f_vec[n,1];
    for (s in 1:nSED){
      for (z in 1:nz) {
	ps[z]<-logpz[n,z]+normal_log(tmp_fvec,tmp_sed[z,s],sigma);	
      }
      increment_log_prob(log_sum_exp(ps));
    }

  }
  //Prior on background 
  bkg_psw ~normal(0,1);
  bkg_pmw ~normal(0,1);
  bkg_plw ~normal(0,1);
}
