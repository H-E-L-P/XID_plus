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

parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f_psw;//source vector
  real bkg_psw;//background
  vector<lower=0.0,upper=1.0>[nsrc] src_f_pmw;//source vector
  real bkg_pmw;//background
  real<lower=0.0,upper=8> sigma_conf_psw;
  real<lower=0.0,upper=8> sigma_conf_pmw;
}

model {
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map


  vector[npix_psw] sigma_tot_psw;
  vector[npix_pmw] sigma_tot_pmw;

  vector[nsrc] f_vec_psw;//vector of source fluxes
  vector[nsrc] f_vec_pmw;//vector of source fluxes
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec_psw[n] <- f_low_lim_psw[n]+(f_up_lim_psw[n]-f_low_lim_psw[n])*src_f_psw[n];
    f_vec_pmw[n] <- f_low_lim_pmw[n]+(f_up_lim_pmw[n]-f_low_lim_pmw[n])*src_f_pmw[n];



  }
  //Prior on background 
  bkg_psw ~normal(bkg_prior_psw,bkg_prior_sig_psw);
  bkg_pmw ~normal(bkg_prior_pmw,bkg_prior_sig_pmw);
 
   
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_psw) {
    db_hat_psw[k] <- bkg_psw;
    sigma_tot_psw[k]<-sqrt(square(sigma_psw[k])+square(sigma_conf_psw));
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec_psw[Col_psw[k]+1];
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <- bkg_pmw;
    sigma_tot_pmw[k]<-sqrt(square(sigma_pmw[k])+square(sigma_conf_pmw));
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*f_vec_pmw[Col_pmw[k]+1];
      }


  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_tot_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_tot_pmw);
    }
