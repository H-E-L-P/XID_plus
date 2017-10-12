//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;            //number of sources
  vector[nsrc] f_low_lim[3];    //upper limit of flux
  vector[nsrc] f_up_lim[3];     //upper limit of flux
  //real f_low_lim[3];
  //real f_up_lim[3];
  vector[nsrc] f_mu[3];         //mu of flux distribution
  vector[nsrc] f_sigma[3];      //sigma of flux distribution
  real bkg_prior[3];            //prior estimate of background
  real bkg_prior_sig[3];        //sigma of prior estimate of background
  //----PSW----
  int<lower=0> npix_psw;        //number of pixels
  int<lower=0> nnz_psw;         //number of non neg entries in A
  vector[npix_psw] db_psw;      //flattened map
  vector[npix_psw] sigma_psw;   //flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_psw] Val_psw;      //non neg values in image matrix
  int Row_psw[nnz_psw];         //Rows of non neg valies in image matrix
  int Col_psw[nnz_psw];         //Cols of non neg values in image matrix
  //----PMW----
  int<lower=0> npix_pmw;        //number of pixels
  int<lower=0> nnz_pmw;         //number of non neg entries in A
  vector[npix_pmw] db_pmw;      //flattened map
  vector[npix_pmw] sigma_pmw;   //flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_pmw] Val_pmw;      //non neg values in image matrix
  int Row_pmw[nnz_pmw];         //Rows of non neg valies in image matrix
  int Col_pmw[nnz_pmw];         //Cols of non neg values in image matrix
  //----PLW----
  int<lower=0> npix_plw;        //number of pixels
  int<lower=0> nnz_plw;         //number of non neg entries in A
  vector[npix_plw] db_plw;      //flattened map
  vector[npix_plw] sigma_plw;   //flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_plw] Val_plw;      //non neg values in image matrix
  int Row_plw[nnz_plw];         //Rows of non neg valies in image matrix
  int Col_plw[nnz_plw];         //Cols of non neg values in image matrix
}

parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f[3];//source vector
  real bkg[3];                               //background
  real<lower=0.0> sigma_conf[3];
}

model {
  vector[npix_psw] db_hat_psw;  //model of map
  vector[npix_pmw] db_hat_pmw;  //model of map
  vector[npix_plw] db_hat_plw;  //model of map

  vector[npix_psw] sigma_tot_psw;
  vector[npix_pmw] sigma_tot_pmw;
  vector[npix_plw] sigma_tot_plw;

  vector[nsrc] f_vec[3];        //vector of source fluxes

  for (i in 1:3){
    // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
    for (n in 1:nsrc) {
      src_f[i,n] ~ normal(f_mu[i,n],f_sigma[i,n]) T[0,1];
      f_vec[i,n] = f_low_lim[i,n]+(f_up_lim[i,n]-f_low_lim[i,n])*src_f[i,n];
    }

    //Prior on background 
    bkg[i] ~normal(bkg_prior[i],bkg_prior_sig[i]);

    //Prior on conf
    sigma_conf[i] ~ normal(0,5);
  }
   
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_psw) {
    db_hat_psw[k] = bkg[1];
    sigma_tot_psw[k] = sqrt(square(sigma_psw[k])+square(sigma_conf[1]));
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] = db_hat_psw[Row_psw[k]+1] + Val_psw[k]*f_vec[1][Col_psw[k]+1];
  }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] =  bkg[2];
    sigma_tot_pmw[k] = sqrt(square(sigma_pmw[k])+square(sigma_conf[2]));
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] = db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*f_vec[2][Col_pmw[k]+1];
  }

  for (k in 1:npix_plw) {
    db_hat_plw[k] = bkg[3];
    sigma_tot_plw[k] = sqrt(square(sigma_plw[k])+square(sigma_conf[3]));
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] = db_hat_plw[Row_plw[k]+1] + Val_plw[k]*f_vec[3][Col_plw[k]+1];
  }

  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_tot_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_tot_pmw);
  db_plw ~ normal(db_hat_plw,sigma_tot_plw);

}
