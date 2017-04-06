//Full Bayesian inference fit  XID
data {
  int<lower=0> nsrc;//number of sources
  //----SCUBA2----
  int<lower=0> npix;//number of pixels
  int<lower=0> nnz; //number of non neg entries in A
  vector[npix] db;//flattened map
  vector[npix] sigma;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior;//prior estimate of background
  real bkg_prior_sig;//sigma of prior estimate of background
  vector[nnz] Val;//non neg values in image matrix
  int Row[nnz];//Rows of non neg valies in image matrix
  int Col[nnz];//Cols of non neg values in image matrix
  vector[nsrc] f_low_lim;//upper limit of flux 
  vector[nsrc] f_up_lim;//upper limit of flux 
  }

parameters {
  vector<lower=0.0,upper=1.0>[nsrc] src_f;//source vector
  real bkg;//background
  
  real<lower=0.0> sigma_conf;
}

model {
  vector[npix] db_hat;//model of map

  vector[npix] sigma_tot;

  vector[nsrc] f_vec;//vector of source fluxes
  
  // Transform to normal space. As I am sampling variable then transforming I don't need a Jacobian adjustment
  for (n in 1:nsrc) {
    f_vec[n] <- f_low_lim[n]+(f_up_lim[n]-f_low_lim[n])*src_f[n];
  }

 //Prior on background 
  bkg ~normal(bkg_prior,bkg_prior_sig);
 
 //Prior on conf
  sigma_conf ~cauchy(0,3);
 
 // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix) {
    db_hat[k] <- bkg;
    sigma_tot[k]<-sqrt(square(sigma[k])+square(sigma_conf));
  }
  for (k in 1:nnz) {
    db_hat[Row[k]+1] <- db_hat[Row[k]+1] + Val[k]*f_vec[Col[k]+1];
      }


  // likelihood of observed map|model map
  db ~ normal(db_hat,sigma_tot);
    }
