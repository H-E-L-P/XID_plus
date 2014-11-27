//Full Bayesian inference fit  XID
data {
  int<lower=0> npix;//number of pixels
  int<lower=0> nsrc;//number of sources
  int<lower=0> nnz; //number of non neg entries in A
  vector[npix] db;//flattened map
  vector[npix] sigma;//flattened uncertianty map (assuming no covariance between pixels)
  real bkg_prior;//prior estimate of background
  real bkg_prior_sig;//sigma of prior estimate of background 
  vector[nnz] Val;//non neg values in image matrix
  int Row[nnz];//Rows of non neg valies in image matrix
  int Col[nnz];//Cols of non neg values in image matrix
}
parameters {
  vector<lower=0.0,upper=300> [nsrc] src_f;//source vector
  real bkg;//background
}

model {
  vector[npix] db_hat;//model of map
  vector[nsrc+1] f_vec;//vector of source fluxes and background


  bkg ~normal(bkg_prior,bkg_prior_sig);
  for (n in 1:nsrc) {
    f_vec[n] <- src_f[n];
  }
  f_vec[nsrc+1] <-bkg;

  #src_f ~cauchy(0,10); // set cauchy distribution for fluxes i.e. expect lower

  for (k in 1:npix) {
    db_hat[k] <- 0;
  }
  for (k in 1:nnz) {
    db_hat[Row[k]+1] <- db_hat[Row[k]+1] + Val[k]*f_vec[Col[k]+1];
      }
  db ~ normal(db_hat,sigma);
    }
