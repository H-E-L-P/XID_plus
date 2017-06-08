//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  vector[nsrc] f_low_lim[1];//upper limit of flux
  vector[nsrc] f_up_lim[1];//upper limit of flux
  real bkg_prior[1];//prior estimate of background
  real bkg_prior_sig[1];//sigma of prior estimate of background
  //----PSW----
  int<lower=0> npix_psw;//number of pixels
  int<lower=0> nnz_psw; //number of non neg entries in A
  vector[npix_psw] db_psw;//flattened map
  vector[npix_psw] sigma_psw;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_psw] Val_psw;//non neg values in image matrix
  int Row_psw[nnz_psw];//Rows of non neg values in image matrix
  int Col_psw[nnz_psw];//Cols of non neg values in image matrix
}
parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f[1];//source vector
  real bkg[1];//background
  real<lower=0.0,upper=0.00001> sigma_conf[1];
}


model {
  vector[npix_psw] db_hat_psw;//model of map



  vector[npix_psw] sigma_tot_psw;


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
  for (k in 1:npix_psw) {
    db_hat_psw[k] <- bkg[1];
    sigma_tot_psw[k]<-sqrt(square(sigma_psw[k])+square(sigma_conf[1]));
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec[1][Col_psw[k]+1];
      }


  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_tot_psw);


    }
