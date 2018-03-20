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
  // ----SED templates----
  int nTemp;
  int nz;
  int nband;
  real SEDs[nTemp,nband,nz];
  //-----------------------
}
parameters {
  vector[2] dist_params[nsrc];//redshift and LIR params
  //vector<lower=0.0>[nband] src_f[nsrc];//vector of source src_fes
  vector[2] mu;
  cov_matrix[2] Sig;
  }

model{

  mu[1] ~normal(0.69,1.5);
  mu[2] ~normal(10.0,3.0);


  for (i in 1:nsrc){
    dist_params[i] ~ multi_normal(mu,Sig);
}
}

