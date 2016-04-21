//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  //----PSW----
  int<lower=0> npix_psw;//number of pixels
  int<lower=0> nnz_psw; //number of non neg entries in A
  vector[npix_psw] db_psw;//flattened map
  vector[npix_psw] sigma_psw;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior_psw;//prior estimate of background
  real bkg_prior_sig_psw;//sigma of prior estimate of background
  vector[nnz_psw] Val_psw;//non neg values in image matrix
  int Row_psw[nnz_psw];//Rows of non neg valies in image matrix
  int Col_psw[nnz_psw];//Cols of non neg values in image matrix
  vector[nsrc] f_low_lim_psw;//upper limit of flux 
  vector[nsrc] f_up_lim_psw;//upper limit of flux 
  //----PMW----
  int<lower=0> npix_pmw;//number of pixels
  int<lower=0> nnz_pmw; //number of non neg entries in A
  vector[npix_pmw] db_pmw;//flattened map
  vector[npix_pmw] sigma_pmw;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior_pmw;//prior estimate of background
  real bkg_prior_sig_pmw;//sigma of prior estimate of background
  vector[nnz_pmw] Val_pmw;//non neg values in image matrix
  int Row_pmw[nnz_pmw];//Rows of non neg valies in image matrix
  int Col_pmw[nnz_pmw];//Cols of non neg values in image matrix
  vector[nsrc] f_low_lim_pmw;//upper limit of flux (in log10)
  vector[nsrc] f_up_lim_pmw;//upper limit of flux (in log10)
  //----PLW----
  int<lower=0> npix_plw;//number of pixels
  int<lower=0> nnz_plw; //number of non neg entries in A
  vector[npix_plw] db_plw;//flattened map
  vector[npix_plw] sigma_plw;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior_plw;//prior estimate of background
  real bkg_prior_sig_plw;//sigma of prior estimate of background
  vector[nnz_plw] Val_plw;//non neg values in image matrix
  int Row_plw[nnz_plw];//Rows of non neg valies in image matrix
  int Col_plw[nnz_plw];//Cols of non neg values in image matrix
  vector[nsrc] f_low_lim_plw;//upper limit of flux 
  vector[nsrc] f_up_lim_plw;//upper limit of flux
  //-----SED-prior--------
  int<lower=0> nz;// number of points in zpdf
  vector[nz] logpz[nsrc];///logpdf for each source
  int<lower=0> nlam;//number of points in SED
  int<lower=0> nSED;//number of SED temps
  real SEDs[nlam,nSED]; // SEDs
  vector[6] sigma;
  vector[nsrc] band7; //ALMA flux at band 7
  vector[nsrc] band7_sig; //ALMA flux uncertianty at band 7


 
}
transformed data {
  vector[6] tmp_sed[nz,nSED];
  for (s in 1:nSED){
      for (z in 1:nz) {
	tmp_sed[z,s,1] <- SEDs[z+2*nz,s]/SEDs[z,s];
	tmp_sed[z,s,2] <- SEDs[z+2*nz,s]/SEDs[z+nz,s];
	tmp_sed[z,s,3] <- SEDs[z+nz,s]/SEDs[z,s];
	tmp_sed[z,s,4] <- SEDs[z+3*nz,s]/SEDs[z+2*nz,s];
	tmp_sed[z,s,5] <- SEDs[z+3*nz,s]/SEDs[z+nz,s];
	tmp_sed[z,s,6] <- SEDs[z+3*nz,s]/SEDs[z,s];

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
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map
  vector[npix_plw] db_hat_plw;//model of map


  vector[npix_psw] sigma_tot_psw;
  vector[npix_pmw] sigma_tot_pmw;
  vector[npix_plw] sigma_tot_plw;

  row_vector[3] f_vec[nsrc];//matrix of source fluxes

  vector[nz] ps;//log probs

  vector[6] tmp_fvec;




  
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {

    f_vec[n,1] <- f_low_lim_psw[n]+(f_up_lim_psw[n]-f_low_lim_psw[n])*src_f_psw[n];
    f_vec[n,2] <- f_low_lim_pmw[n]+(f_up_lim_pmw[n]-f_low_lim_pmw[n])*src_f_pmw[n];
    f_vec[n,3] <- f_low_lim_plw[n]+(f_up_lim_plw[n]-f_low_lim_plw[n])*src_f_plw[n];
    tmp_fvec[1] <- f_vec[n,3]/f_vec[n,1];
    tmp_fvec[2] <- f_vec[n,3]/f_vec[n,2];
    tmp_fvec[3] <- f_vec[n,2]/f_vec[n,1];
    tmp_fvec[4] <- band7[n]/f_vec[n,3];
    tmp_fvec[5] <- band7[n]/f_vec[n,2];
    tmp_fvec[6] <- band7[n]/f_vec[n,1];


    for (s in 1:nSED){
      for (z in 1:nz) {
	    ps[z]<-logpz[n,z]+normal_log(tmp_fvec,tmp_sed[z,s],sigma);
      }
      increment_log_prob(log_sum_exp(ps));
    }

  }
  //Prior on background 
  bkg_psw ~normal(bkg_prior_psw,bkg_prior_sig_psw);
  bkg_pmw ~normal(bkg_prior_pmw,bkg_prior_sig_pmw);
  bkg_plw ~normal(bkg_prior_plw,bkg_prior_sig_plw);

  //Prior on sigma
  sigma_conf_psw ~ cauchy(0, 5);

  sigma_conf_pmw ~ cauchy(0, 5);

  sigma_conf_plw ~ cauchy(0, 5);



 
   
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_psw) {
    db_hat_psw[k] <- bkg_psw;
    sigma_tot_psw[k]<-sqrt(square(sigma_psw[k])+square(sigma_conf_psw));
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec[Col_psw[k]+1,1];
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <- bkg_pmw;
    sigma_tot_pmw[k]<-sqrt(square(sigma_pmw[k])+square(sigma_conf_pmw));
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*f_vec[Col_pmw[k]+1,2];
      }

  for (k in 1:npix_plw) {
    db_hat_plw[k] <- bkg_plw;
    sigma_tot_plw[k]<-sqrt(square(sigma_plw[k])+square(sigma_conf_plw));
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] <- db_hat_plw[Row_plw[k]+1] + Val_plw[k]*f_vec[Col_plw[k]+1,3];
      }

  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_tot_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_tot_pmw);
  db_plw ~ normal(db_hat_plw,sigma_tot_plw);



  
  // As actual maps are mean subtracted, requires a Jacobian adjustment
  //db_psw <- db_obs_psw - mean(db_obs_psw)
  //increment_log_prob(log((size(db_obs_psw)-1)/size(db_obs_psw)))
  //db_pmw <- db_obs_pmw - mean(db_obs_pmw)
  //increment_log_prob(log((size(db_obs_pmw)-1)/size(db_obs_pmw)))
  //db_plw <- db_obs_plw - mean(db_obs_plw)
  //increment_log_prob(log((size(db_obs_plw)-1)/size(db_obs_plw)))
    }
