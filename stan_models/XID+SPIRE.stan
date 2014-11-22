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

}
parameters {
  vector<lower=0.0,upper=300> [nsrc] src_f_psw;//source vector
  real bkg_psw;//background
  vector<lower=0.0,upper=300> [nsrc] src_f_pmw;//source vector
  real bkg_pmw;//background
  vector<lower=0.0,upper=300> [nsrc] src_f_plw;//source vector
  real bkg_plw;//background

}

model {
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map
  vector[npix_plw] db_hat_plw;//model of map

  vector[nsrc+1] f_vec_psw;//vector of source fluxes and background
  vector[nsrc+1] f_vec_pmw;//vector of source fluxes and background
  vector[nsrc+1] f_vec_plw;//vector of source fluxes and background



  bkg_psw ~normal(bkg_prior_psw,bkg_prior_sig_psw);
  bkg_pmw ~normal(bkg_prior_pmw,bkg_prior_sig_pmw);
  bkg_plw ~normal(bkg_prior_plw,bkg_prior_sig_plw);
 
  for (n in 1:nsrc) {
    f_vec_psw[n] <- src_f_psw[n];
    f_vec_pmw[n] <- src_f_pmw[n];
    f_vec_plw[n] <- src_f_plw[n];


  }
  f_vec_psw[nsrc+1] <-bkg_psw;
  f_vec_pmw[nsrc+1] <-bkg_pmw;
  f_vec_plw[nsrc+1] <-bkg_plw;


  #src_f ~cauchy(0,10); // set cauchy distribution for fluxes i.e. expect lower

  for (k in 1:npix_psw) {
    db_hat_psw[k] <- 0;
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec_psw[Col_psw[k]+1];
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <- 0;
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*f_vec_pmw[Col_pmw[k]+1];
      }

  for (k in 1:npix_plw) {
    db_hat_plw[k] <- 0;
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] <- db_hat_plw[Row_plw[k]+1] + Val_plw[k]*f_vec_plw[Col_plw[k]+1];
      }


  db_psw ~ normal(db_hat_psw,sigma_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_pmw);
  db_plw ~ normal(db_hat_plw,sigma_plw);

    }
