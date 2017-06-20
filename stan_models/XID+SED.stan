functions {
int intFloor(int leftStart, int rightStart, real iReal)
{
  // This is absurd. Use bisection algorithm to find int floor.
  int left;
  int right;

  left <- leftStart;
  right <- rightStart;

  while((left + 1) < right) {
    int mid;
    // print("left, right, mid, i, ", left, ", ", right, ", ", mid, ", ", iReal);
    mid <- left + (right - left) / 2;
    if(iReal < mid) {
      right <- mid;
    }
    else {
      left <- mid;
    }
  }
  return left;
}

// Interpolate arr using a non-integral index i
// Note: 1 <= i <= length(arr)
real interpolateLinear(real[] arr, real i)
{
  int iLeft;
  real valLeft;
  int iRight;
  real valRight;

  // print("interpolating ", i);

  // Get i, value at left. If exact time match, then return value.
  iLeft <- intFloor(1, size(arr), i);
  valLeft <- arr[iLeft];
  if(iLeft == i) {
    return valLeft;
  }

  // Get i, value at right.
  iRight <- iLeft + 1;
  valRight <- arr[iRight];

  // Linearly interpolate between values at left and right.
  return valLeft + (valRight - valLeft) * (i - iLeft);
}


}
data
{
  int<lower=0> nsrc;//number of sources
  real<lower=0.0> z[nsrc];
  real<lower=0.0> z_sig[nsrc];
  // ----SED templates----
    int nTemp;
    int nz;
  int nband;
    real SEDs[nTemp,nband,nz];
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
  //----other photometry----
  vector[nband-3] flux_obs[nsrc];//in mJy
  vector[nband-3] flux_sig[nsrc];//in mJy
}

parameters {
  real<lower=8, upper=14> Nbb[nsrc];
  //real<lower=0.001,upper=7> z[nsrc];
  vector<lower=0.0>[nband] flux[nsrc];//vector of source fluxes
  real<upper=0.0> bkg_psw;//background
  real<upper=0.0> bkg_pmw;//background
  real<upper=0.0> bkg_plw;//background
  real<lower=0.0> sigma_conf_psw;
  real<lower=0.0> sigma_conf_pmw;
  real<lower=0.0> sigma_conf_plw;
}


model{
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map
  vector[npix_plw] db_hat_plw;//model of map


  vector[npix_psw] sigma_tot_psw;
  vector[npix_pmw] sigma_tot_pmw;
  vector[npix_plw] sigma_tot_plw;
  for (i in 1:nsrc){
    vector[nTemp] ps;//log prob
    for (t in 1:nTemp){
        vector[nband] f_tmp;
	for (b in 1:nband){
        f_tmp[b]=pow(10.0,Nbb[i])*interpolateLinear(SEDs[t,b], z[i]*100.0);
	}
	//print(f_tmp)
        ps[t]<-normal_lpdf(flux[i]|f_tmp,f_tmp/5.0);   
    }
    target+=log_sum_exp(ps);
    //flux_obs[i]~normal(flux[i,4:nband],flux_sig[i]);
    //z~normal(z_mean,z_sig);
  }

 //Prior on background 
  bkg_psw ~normal(bkg_prior_psw,bkg_prior_sig_psw);
  bkg_pmw ~normal(bkg_prior_pmw,bkg_prior_sig_pmw);
  bkg_plw ~normal(bkg_prior_plw,bkg_prior_sig_plw); 

 //Prior on conf
  sigma_conf_psw ~normal(0,5);
  sigma_conf_pmw ~normal(0,5);
  sigma_conf_plw ~normal(0,5);
   
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_psw) {
    db_hat_psw[k] <- bkg_psw;
    sigma_tot_psw[k]<-sqrt(square(sigma_psw[k])+square(sigma_conf_psw));
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*flux[Col_psw[k]+1,1];
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <-  bkg_pmw;
    sigma_tot_pmw[k]<-sqrt(square(sigma_pmw[k])+square(sigma_conf_pmw));
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*flux[Col_pmw[k]+1,2];
      }

  for (k in 1:npix_plw) {
    db_hat_plw[k] <- bkg_plw;
    sigma_tot_plw[k]<-sqrt(square(sigma_plw[k])+square(sigma_conf_plw));
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] <- db_hat_plw[Row_plw[k]+1] + Val_plw[k]*flux[Col_plw[k]+1,3];
      }
  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_tot_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_tot_pmw);
  db_plw ~ normal(db_hat_plw,sigma_tot_plw);

}
generated quantities {

matrix[nsrc,nTemp] p;
for (i in 1:nsrc){
    vector[nTemp] p_raw;
     for (t in 1:nTemp){
        vector[nband] f_tmp;
	for (b in 1:nband) {
        f_tmp[b]=pow(10.0,Nbb[i])*interpolateLinear(SEDs[t,b], z[i]*100.0);
	}
        p_raw[t] = (1.0/nTemp)*exp(normal_lpdf(flux[i]|f_tmp,f_tmp/5.0));
     }
     for (t in 1:nTemp){
     p[i,t]=p_raw[t]/sum(p_raw);
     }
 }
}
