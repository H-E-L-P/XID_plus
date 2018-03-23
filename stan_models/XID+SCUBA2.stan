//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  vector[nsrc] f_low_lim[1];//upper limit of flux
  vector[nsrc] f_up_lim[1];//upper limit of flux
  real bkg_prior[1];//prior estimate of background
  real bkg_prior_sig[1];//sigma of prior estimate of background
  //----S2B----
  int<lower=0> npix_S2b;//number of pixels
  int<lower=0> nnz_S2b; //number of non neg entries in A
  vector[npix_S2b] db_S2b;//flattened map
  vector[npix_S2b] sigma_S2b;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_S2b] Val_S2b;//non neg values in image matrix
  int Row_S2b[nnz_S2b];//Rows of non neg values in image matrix
  int Col_S2b[nnz_S2b];//Cols of non neg values in image matrix
}


parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f[1];//source vector
  real bkg[1];//background
  real<lower=0.0> sigma_conf[1];
}


model {
  vector[npix_S2b] db_hat_S2b;//model of map

  vector[npix_S2b] sigma_tot_S2b;

  vector[nsrc] f_vec[1];//vector of source fluxes

  for (i in 1:1){
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec[i,n] <- f_low_lim[i,n]+(f_up_lim[i,n]-f_low_lim[i,n])*src_f[i,n];



  }

 //Prior on background
  bkg[i] ~normal(bkg_prior[i],bkg_prior_sig[i]);

 //Prior on conf
  sigma_conf[i] ~normal(0,5);
  }

  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_S2b) {
    db_hat_S2b[k] <- bkg[1];
    sigma_tot_S2b[k]<-sqrt(square(sigma_S2b[k])+square(sigma_conf[1]));
  }
  for (k in 1:nnz_S2b) {
    db_hat_S2b[Row_S2b[k]+1] <- db_hat_S2b[Row_S2b[k]+1] + Val_S2b[k]*f_vec[1][Col_S2b[k]+1];
      }

  // likelihood of observed map|model map
  db_S2b ~ normal(db_hat_S2b,sigma_tot_S2b);

    }
