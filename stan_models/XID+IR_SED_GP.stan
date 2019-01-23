
functions {
  functions {
  vector gp_pred_lpdf(real[] x2,
                     vector y1, real[] x1,
                     real alpha, real rho, real sigma, real delta,vector src_f) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho)
                         + diag_matrix(rep_vector(square(sigma), N1));
      matrix[N1, N1] L_K = cholesky_decompose(K);

      vector[N1] L_K_div_y1 = mdivide_left_tri_low(L_K, y1);
      vector[N1] K_div_y1 = mdivide_right_tri_low(L_K_div_y1', L_K)';
      matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
      vector[N2] f2_mu = (k_x1_x2' * K_div_y1);
      matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, alpha, rho) - v_pred' * v_pred
                              + diag_matrix(rep_vector(delta, N2));
    }
    return multi_normal_lpdf(src_f|f2_mu,cov_f2);
  }
}
data
{
  int<lower=0> nsrc;//number of sources
  real bkg_prior[6];//prior estimate of background
  real bkg_prior_sig[6];//sigma of prior estimate of background
  real conf_prior_sig[6];
  real<lower=0.0> z_median[nsrc];
  real<lower=0.0> z_sig[nsrc];
  //----PSW----
  int<lower=0> npix_psw;//number of pixels
  int<lower=0> nnz_psw; //number of non neg entries in A
  vector[npix_psw] db_psw;//flattened map
  vector[npix_psw] sigma_psw;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_psw] Val_psw;//non neg values in image matrix
  int Row_psw[nnz_psw];//Rows of non neg values in image matrix
  int Col_psw[nnz_psw];//Cols of non neg values in image matrix
  //----PMW----
  int<lower=0> npix_pmw;//number of pixels
  int<lower=0> nnz_pmw; //number of non neg entries in A
  vector[npix_pmw] db_pmw;//flattened map
  vector[npix_pmw] sigma_pmw;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_pmw] Val_pmw;//non neg values in image matrix
  int Row_pmw[nnz_pmw];//Rows of non neg valies in image matrix
  int Col_pmw[nnz_pmw];//Cols of non neg values in image matrix
  //----PLW----
  int<lower=0> npix_plw;//number of pixels
  int<lower=0> nnz_plw; //number of non neg entries in A
  vector[npix_plw] db_plw;//flattened map
  vector[npix_plw] sigma_plw;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_plw] Val_plw;//non neg values in image matrix
  int Row_plw[nnz_plw];//Rows of non neg valies in image matrix
  int Col_plw[nnz_plw];//Cols of non neg values in image matrix
  //----MIPS------
  int<lower=0> npix_mips24;//number of pixels
  int<lower=0> nnz_mips24; //number of non neg entries in A
  vector[npix_mips24] db_mips24;//flattened map
  vector[npix_mips24] sigma_mips24;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_mips24] Val_mips24;//non neg values in image matrix
  int Row_mips24[nnz_mips24];//Rows of non neg valies in image matrix
  int Col_mips24[nnz_mips24];//Cols of non neg values in image matrix
    //----PACS green----
  int<lower=0> npix_pacs100;//number of pixels
  int<lower=0> nnz_pacs100; //number of non neg entries in A
  vector[npix_pacs100] db_pacs100;//flattened map
  vector[npix_pacs100] sigma_pacs100;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_pacs100] Val_pacs100;//non neg values in image matrix
  int Row_pacs100[nnz_pacs100];//Rows of non neg values in image matrix
  int Col_pacs100[nnz_pacs100];//Cols of non neg values in image matrix
    //----PACS red----
  int<lower=0> npix_pacs160;//number of pixels
  int<lower=0> nnz_pacs160; //number of non neg entries in A
  vector[npix_pacs160] db_pacs160;//flattened map
  vector[npix_pacs160] sigma_pacs160;//flattened uncertianty map (assuming no covariance between pixels)
  vector[nnz_pacs160] Val_pacs160;//non neg values in image matrix
  int Row_pacs160[nnz_pacs160];//Rows of non neg values in image matrix
  int Col_pacs160[nnz_pacs160];//Cols of non neg values in image matrix

  // ----SED templates----
  int nTemp;
  int nz;
  int nband;
  real SEDs[nTemp,nband,nz];
  //real SEDs_sig[nTemp,nz];
  //-----------------------

}

parameters {
  real<lower=8, upper=14> Nbb[nsrc];
  real<lower=0.001,upper=7> z[nsrc];
  vector<lower=-5,upper=4>[nsrc] src_f[nband];//vector of source src_fes
  real bkg[nband];//background

}
transformed parameters{
  real<lower=0.0> sigma_conf[nband];
  for (i in 1:nband){
    sigma_conf[i]=0.0;
}
}



model{
  vector[npix_psw] db_hat_psw;//model of map
  vector[npix_pmw] db_hat_pmw;//model of map
  vector[npix_plw] db_hat_plw;//model of map
  vector[npix_mips24] db_hat_mips24;//model of map
  vector[npix_pacs100] db_hat_pacs100;//model of map
  vector[npix_pacs160] db_hat_pacs160;//model of map



  vector[npix_psw] sigma_tot_psw;
  vector[npix_pmw] sigma_tot_pmw;
  vector[npix_plw] sigma_tot_plw;
  vector[npix_mips24] sigma_tot_mips24;
  vector[npix_pacs100] sigma_tot_pacs100;
  vector[npix_pacs160] sigma_tot_pacs160;

  vector[nband] ps;//log prob


  for (i in 1:nband){
  //Prior on background
  bkg[i] ~normal(bkg_prior[i],bkg_prior_sig[i]);

 //Prior on conf
  //sigma_conf[i] ~normal(0,conf_prior_sig[i]);
  }




    z~normal(z_median,z_sig);
    for (b in 1:nband){
    for (t in 1:nTemp){
        ps[t]<-gp_pred_lpdf(z,SEDs[t,b],red, alpha, rho, sigma, 1e-10, src_f[b]-Nbb);
	}
    target+=log_sum_exp(ps)
    }




   
  // Create model maps (i.e. db_hat = A*f) using sparse multiplication
  for (k in 1:npix_mips24) {
    db_hat_mips24[k] <- bkg[1];
    sigma_tot_mips24[k]<-sqrt(square(sigma_mips24[k])+square(sigma_conf[1]));
  }
  for (k in 1:nnz_mips24) {
    db_hat_mips24[Row_mips24[k]+1] <- db_hat_mips24[Row_mips24[k]+1] + Val_mips24[k]*pow(10.0,src_f[1][Col_mips24[k]+1]);
      }

  for (k in 1:npix_pacs100) {
    db_hat_pacs100[k] <- bkg[2];
    sigma_tot_pacs100[k]<-sqrt(square(sigma_pacs100[k])+square(sigma_conf[2]));
  }
  for (k in 1:nnz_pacs100) {
    db_hat_pacs100[Row_pacs100[k]+1] <- db_hat_pacs100[Row_pacs100[k]+1] + Val_pacs100[k]*pow(10.0,src_f[2][Col_pacs100[k]+1]);
      }

  for (k in 1:npix_pacs160) {
    db_hat_pacs160[k] <- bkg[3];
    sigma_tot_pacs160[k]<-sqrt(square(sigma_pacs160[k])+square(sigma_conf[3]));
  }
  for (k in 1:nnz_pacs160) {
    db_hat_pacs160[Row_pacs160[k]+1] <- db_hat_pacs160[Row_pacs160[k]+1] + Val_pacs160[k]*pow(10.0,src_f[3][Col_pacs160[k]+1]);
      }


  for (k in 1:npix_psw) {
    db_hat_psw[k] <- bkg[4];
    sigma_tot_psw[k]<-sqrt(square(sigma_psw[k])+square(sigma_conf[4]));
  }
  for (k in 1:nnz_psw) {
    db_hat_psw[Row_psw[k]+1] <- db_hat_psw[Row_psw[k]+1] + Val_psw[k]*pow(10.0,src_f[4][Col_psw[k]+1]);
      }

  for (k in 1:npix_pmw) {
    db_hat_pmw[k] <-  bkg[5];
    sigma_tot_pmw[k]<-sqrt(square(sigma_pmw[k])+square(sigma_conf[5]));
  }
  for (k in 1:nnz_pmw) {
    db_hat_pmw[Row_pmw[k]+1] <- db_hat_pmw[Row_pmw[k]+1] + Val_pmw[k]*pow(10.0,src_f[5][Col_pmw[k]+1]);
      }

  for (k in 1:npix_plw) {
    db_hat_plw[k] <- bkg[6];
    sigma_tot_plw[k]<-sqrt(square(sigma_plw[k])+square(sigma_conf[6]));
  }
  for (k in 1:nnz_plw) {
    db_hat_plw[Row_plw[k]+1] <- db_hat_plw[Row_plw[k]+1] + Val_plw[k]*pow(10.0,src_f[6][Col_plw[k]+1]);
      }



  // likelihood of observed map|model map
  db_psw ~ normal(db_hat_psw,sigma_tot_psw);
  db_pmw ~ normal(db_hat_pmw,sigma_tot_pmw);
  db_plw ~ normal(db_hat_plw,sigma_tot_plw);
  db_mips24 ~ normal(db_hat_mips24,sigma_tot_mips24);
  db_pacs100 ~ normal(db_hat_pacs100,sigma_tot_pacs100);
  db_pacs160 ~ normal(db_hat_pacs160,sigma_tot_pacs160);




}
