//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  //----SCUBA2----
  int<lower=0> npix_sc2;//number of pixels
  int<lower=0> nnz_sc2; //number of non neg entries in A
  vector[npix_sc2] db_sc2;//flattened map
  vector[npix_sc2] sigma_sc2;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior_sc2;//prior estimate of background
  real bkg_prior_sig_sc2;//sigma of prior estimate of background
  vector[nnz_sc2] Val_sc2;//non neg values in image matrix
  int Row_sc2[nnz_sc2];//Rows of non neg valies in image matrix
  int Col_sc2[nnz_sc2];//Cols of non neg values in image matrix
  vector[nsrc] f_low_lim_sc2;//upper limit of flux 
  vector[nsrc] f_up_lim_sc2;//upper limit of flux 
  }

parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f_sc2;//source vector
  real bkg_sc2;//background
  
  real<lower=0.0> sigma_conf_sc2;
}

model {
  vector[npix_sc2] db_hat_sc2;//model of map

  vector[npix_sc2] sigma_tot_sc2;

  vector[nsrc] f_vec_sc2;//vector of source fluxes
  
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec_sc2[n] <- f_low_lim_sc2[n]+(f_up_lim_sc2[n]-f_low_lim_sc2[n])*src_f_sc2[n];
  }

 //Prior on background 
  bkg_sc2 ~normal(bkg_prior_sc2,bkg_prior_sig_sc2);
 
 //Prior on conf
  sigma_conf_sc2 ~cauchy(0,3);
 
 // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_sc2) {
    db_hat_sc2[k] <- bkg_sc2;
    sigma_tot_sc2[k]<-sqrt(square(sigma_sc2[k])+square(sigma_conf_sc2));
  }
  for (k in 1:nnz_sc2) {
    db_hat_sc2[Row_sc2[k]+1] <- db_hat_sc2[Row_sc2[k]+1] + Val_sc2[k]*f_vec_sc2[Col_sc2[k]+1];
      }


  // likelihood of observed map|model map
  db_sc2 ~ normal(db_hat_sc2,sigma_tot_sc2);
    }
