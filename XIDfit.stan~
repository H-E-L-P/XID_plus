//Full Bayesian inference fit on XID sims
data {
  int<lower=0> npix;//number of pixels
  int<lower=0> nsrc;//number of sources
  int<lower=0> nnz; //number of non neg entries in A
  vector[npix] db;//flattened map
  vector[npix] sigma;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz] Val;//non neg values in image matrix
  int Row[nnz];//Rows of non neg valies in image matrix
  int Col[nnz];//Cols of non neg values in image matrix
}
parameters {
  vector<lower=0.0> [nsrc] src_f;//source vector
}

model {
  vector[nsrc] log10src_f;
  vector[npix] db_hat;
  log10src_f <- log(src_f)/log(10.0);
  log10src_f ~ normal(0,1);

  for (k in 1:npix) {
    db_hat[k] <- 0.0;
  }
  for (k in 1:nnz) {
    db_hat[Row[k]+1] <- db_hat[Row[k]+1] + Val[k]*src_f[Col[k]+1];
      }
  db ~ normal(db_hat,sigma);
    }
