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
  vector[nsrc] f_up_lim_psw;//upper limit of flux (in log10)
  int<lower=0> nnz_sig_conf_psw_tot;// total number of 
  int Row_sig_conf_psw[nnz_sig_conf_psw_tot];//row of sigma
  int Col_sig_conf_psw[nnz_sig_conf_psw_tot];//col of sigma
  int Val_sig_conf_psw[nnz_sig_conf_psw_tot];//which sigma
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
  vector[nsrc] f_up_lim_plw;//upper limit of flux (in log10)

}

transformed data {
  matrix[npix_psw,npix_psw] Corr_conf;
  matrix[npix_psw,npix_psw] L;
  for (j in 1:npix_psw){
    for (k in 1:npix_psw){
      Corr_conf[j,k] <- 0.0;
    }
  }
  for (k in 1:nnz_sig_conf_psw_tot) {
    Corr_conf[Row_sig_conf_psw[k],Col_sig_conf_psw[k]] <-Val_sig_conf_psw[k];
  }
  L <- cholesky_decompose(Corr_conf);  
}
parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f_psw;//source vector
  real bkg_psw;//background
  vector<lower=0.0,upper=1.0>[nsrc] src_f_pmw;//source vector
  real bkg_pmw;//background
  vector<lower=0.0,upper=1.0>[nsrc] src_f_plw;//source vector
  real bkg_plw;//background
  real<lower=2.0,upper=6.0> sigma2_conf;//sigma

}


model {
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map
  vector[npix_plw] db_hat_plw;//model of map
  matrix[npix_psw,npix_psw] Sig_tot;//


  vector[nsrc] f_vec_psw;//vector of source fluxes and background
  vector[nsrc] f_vec_pmw;//vector of source fluxes and background
  vector[nsrc] f_vec_plw;//vector of source fluxes

  //Prior on background 
  bkg_psw ~normal(bkg_prior_psw,bkg_prior_sig_psw);
  bkg_pmw ~normal(bkg_prior_pmw,bkg_prior_sig_pmw);
  bkg_plw ~normal(bkg_prior_plw,bkg_prior_sig_plw);
 
  

  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec_psw[n] <- pow(10.0,-2.0+(f_up_lim_psw[n]+2.0)*src_f_psw[n]);
    f_vec_pmw[n] <- pow(10.0,-2.0+(f_up_lim_pmw[n]+2.0)*src_f_pmw[n]);
    f_vec_plw[n] <- pow(10.0,-2.0+(f_up_lim_plw[n]+2.0)*src_f_plw[n]);


  }
  Sig_tot<- sigma2_conf*multiply_lower_tri_self_transpose(L);
  
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_psw) {
    db_hat_psw[k] <- bkg_psw;
    //Sig_tot[k] <-0.0;
    Sig_tot[k,k] <- Sig_tot[k,k]+pow(sigma_psw[k],2);
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec_psw[Col_psw[k]+1];
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <- bkg_pmw;
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*f_vec_pmw[Col_pmw[k]+1];
      }

  for (k in 1:npix_plw) {
    db_hat_plw[k] <- bkg_plw;
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] <- db_hat_plw[Row_plw[k]+1] + Val_plw[k]*f_vec_plw[Col_plw[k]+1];
      }



  // likelihood of observed map|model map
  db_psw ~ multi_normal(db_hat_psw,Sig_tot);
  db_pmw ~ normal(db_hat_pmw,sigma_pmw);
  db_plw ~ normal(db_hat_plw,sigma_plw);


  // As actual maps are mean subtracted, requires a Jacobian adjustment
  //db_psw <- db_obs_psw - mean(db_obs_psw)
  //increment_log_prob(log((size(db_obs_psw)-1)/size(db_obs_psw)))
  //db_pmw <- db_obs_pmw - mean(db_obs_pmw)
  //increment_log_prob(log((size(db_obs_pmw)-1)/size(db_obs_pmw)))
  //db_plw <- db_obs_plw - mean(db_obs_plw)
  //increment_log_prob(log((size(db_obs_plw)-1)/size(db_obs_plw)))
    }
