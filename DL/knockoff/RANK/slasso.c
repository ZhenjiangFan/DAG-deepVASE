
/* headers for R related definitions */
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
  
/* local definitions of functions */
/* max */
#ifndef max
#define max(a, b) (a > b ? a : b)
#endif
/* min */
#ifndef min
#define min(a, b) (a < b ? a : b)
#endif 
/* soft-thresholding, but slower to use this function compared to placing 
the expression directly in the code */
#ifndef softshri
#define softshri(beta, lam) (max(0, beta - lam) - max(0, -beta - lam))
#endif

/* Lasso implemented with coordinate descent */
/* each column of n x p design matrix X is rescaled to have unit L2-norm */

/* use function type void to return values by arguments only */
/* use only pointers *something (like addresses for initial elements) for arguments */
/* the last few arguments are for outputs */
  
void slasso(double *X, int *pnrow, int *pncol, double *y, double *plam, int *pmaxite, double *ptol, double *betap)
{
  /* n = *pnrow (value pointed to by the pointer pnrow) is the same as n = pnrow[0] */
  int n = *pnrow, p = *pncol, maxite = *pmaxite;
  double lam = *plam, tol = *ptol;
  /* need to initialize all variables */
  /* use 0 for integers and 0.0 for doubles */
  int i, j, k = 0;
  /* initialize betap = 0 */
  //for (j = 0; j < p; j++) betap[j] = 0.0;
  /* always specify the length as in double betap_old[p], instead of double *betap_old */
  /* define a new vector */
  double betap_old[p];
  /* initialize betap_old = 0 */
  for (j = 0; j < p; j++) betap_old[j] = 0.0;
  double yhat[n];
  for (i = 0; i < n; i++) yhat[i] = 0.0;
  int mm[p];
  for (j = 0; j < p; j++) mm[j] = 0;
  int m[p];
  for (j = 0; j < p; j++) m[j] = 0;
  int nin = 0;
  
  int inner = 0;
  double dlx = 0.0;
  double prod = 0.0;
  double diff_beta = 0.0;
  double diffabs = 0.0;
  while (inner < maxite) {
    dlx = 0.0;
    for (j = 0; j < p; j++) {
      prod = 0.0;
      for (i = 0; i < n; i++) {
        prod += X[j*n + i]*(y[i] - yhat[i]);
      }
      //betap[j] = softshri(prod + betap_old[j], lam); /* this option is slower than below */
      //printf("%f", betap[0]);
      betap[j] = max(0.0, prod + betap_old[j] - lam) - max(0.0, -(prod + betap_old[j]) - lam);
      diff_beta = betap_old[j] - betap[j];
      diffabs = fabs(diff_beta);
      if (diffabs > 0.0) {
        for (i = 0; i < n; i++) yhat[i] += -X[j*n + i]*diff_beta;
        betap_old[j] = betap[j];
        if (mm[j] == 0) {
          nin += 1;
          mm[j] = nin;
          m[nin - 1] = j; 
        }
        dlx = max(dlx, diffabs);
      }
    }
    if (dlx < tol) break;
    
    while (1) {
      dlx = 0.0;
      for (k = 0; k < nin; k++) {
        j = m[k];
        prod = 0.0;
        for (i = 0; i < n; i++) {
          prod += X[j*n + i]*(y[i] - yhat[i]);
        }
        //betap[j] = softshri(prod + betap_old[j], lam); /* this option is slower than below */
        betap[j] = max(0.0, prod + betap_old[j] - lam) - max(0.0, -(prod + betap_old[j]) - lam);
        diff_beta = betap_old[j] - betap[j];
        diffabs = fabs(diff_beta);
        if (diffabs > 0.0) {
          for (i = 0; i < n; i++) yhat[i] += -X[j*n + i]*diff_beta;
          betap_old[j] = betap[j];
          dlx = max(dlx, diffabs);
        }
      }
      if (dlx < tol) break;
    }
    inner += 1;
  }
}
