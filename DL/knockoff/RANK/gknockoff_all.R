# This include the Knockoff and Knockoff plus fliter to computer FDR and Power
# This master R file includes all the R functions for implementing the knockoff filter and 
# knockoff filter plus in high-dimensional linear regression with precision matrix of 
# covariates estimated by the innovated scalable efficient estimation (ISEE) procedure 
# introduced in Fan  and Lv (2015).
#
# The ISEE framework has a range of potential applications involving precision 
# (innovated data) matrix including:
#   Linear and nonlinear classifications;
#   Dimension reduction; 
#   Portfolio management;
#   Feature screening and simultaneous inference.
#
# Written by 
# Yingying Fan (fanyingy@marshall.usc.edu) and Jinchi Lv (jinchilv@marshall.usc.edu)
# USC, April 10, 2014
#
# Reference:
# Fan, Y. and Lv, J. (2015). Innovated scalable efficient estimation in ultra-large 
# Gaussian graphical models. The Annals of Statistics, to appear.
#
# List of R functions for ISEE:
# Part I: main functions
#   1) gknockoff: main function of graphical knockoff filter
#   2) gknockoffX: function generating the knockoff design matrix
#   3) slassotf: function implementing tuning-free sparse regression with scaled Lasso
# Part II: additional functions
#   1) fdr: calculating fdr
#   2) pow: calculating power
#

###########################################################################
# Part I: main functions
# Part I 1) gknockoff: graphical knockoff filter
# Input:
#   X:   n x p design matrix
#   y:   n x 1 response vector
#   q:   nominal FDR level
# Output: 
#   S:   set of discovered variables

gknockoff <- function(X, y, q, regfactor = "log", npermu = 5, sis.use = 0, bia.cor = 0){
  # initialization of ISEE parameters
  # regfactor = "log"  # or "one", "sqrt"
  # npermu = 1         # or >= 2
  # sis.use = 0        # or 1, whether to use SIS for screening
  # bia.cor = 0        # or 1, whether to apply bias correction for ISEE
  
  n = nrow(X)
  p = ncol(X)
  
  # ISEE
  obj = isee(X, regfactor, npermu, sis.use, bia.cor) 
  if (bia.cor == 1){
    # ISEE with bias correction
    Omega = obj$Omega.isee.c
  } else {
    # ISEE with no bias correction
    Omega = obj$Omega.isee
  }
  
  # generate the knockoff design matrix
  Xnew = gknockoffX(X, Omega)
  
  # apply self-tuned scaled Lasso
  obj1 = slassotf(Xnew, y, regfactor, sis.use = 0)
  # beta vector under aligned scale
  betap.stan = obj1$betap.stan
  
  # Calculate knockoff test statistics W_j
  W = c(rep(0, p))
  for (j in 1:p) {
    W[j] = abs(betap.stan[j]) - abs(betap.stan[j+p])
  }
  
  # find the knockoff threshold T
  t = sort(c(0, abs(W)))
  ratio = c(rep(0, p))
  for (j in 1:p) {
    ratio[j] = sum(W <= -t[j])/max(1, sum(W >= t[j]))
  }
  id = which(ratio <= q)[1]
  if(length(id) == 0){
    T = Inf
  } else {
    T = t[id]
  }
  
  # set of discovered variables
  S = which(W >= T)
  
  return(S)
}
###########################################################################

###########################################################################
# Part I: main functions
# Part I 1) gknockoff plus: graphical knockoff plus filter
# Input:
#   X:   n x p design matrix
#   y:   n x 1 response vector
#   q:   nominal FDR level
# Output: 
#   S_plus:   set of discovered variables

gknockoff_plus <- function(X, y, q, regfactor = "log", npermu = 5, sis.use = 0, bia.cor = 0){
  # initialization of ISEE parameters
  # regfactor = "log"  # or "one", "sqrt"
  # npermu = 1         # or >= 2
  # sis.use = 0        # or 1, whether to use SIS for screening
  # bia.cor = 0        # or 1, whether to apply bias correction for ISEE
  
  n = nrow(X)
  p = ncol(X)
  
  # ISEE
  obj = isee(X, regfactor, npermu, sis.use, bia.cor) 
  if (bia.cor == 1){
    # ISEE with bias correction
    Omega = obj$Omega.isee.c
  } else {
    # ISEE with no bias correction
    Omega = obj$Omega.isee
  }
  
  # generate the knockoff design matrix
  Xnew = gknockoffX(X, Omega)
  
  # apply self-tuned scaled Lasso
  obj1 = slassotf(Xnew, y, regfactor, sis.use = 0)
  # beta vector under aligned scale
  betap.stan = obj1$betap.stan
  
  # Calculate knockoff test statistics W_j
  W = c(rep(0, p))
  for (j in 1:p) {
    W[j] = abs(betap.stan[j]) - abs(betap.stan[j+p])
  }
  
  # find the knockoff threshold T
  t = sort(c(0, abs(W)))
  ratio = c(rep(0, p))
  for (j in 1:p) {
    ratio[j] = (1+sum(W <= -t[j]))/max(1, sum(W >= t[j]))
  }
  id = which(ratio <= q)[1]
  if(length(id) == 0){
    T = Inf
  } else {
    T = t[id]
  }
  
  # set of discovered variables
  S_plus = which(W >= T)
  
  return(S_plus)
}
###########################################################################


###########################################################################
# Part I: main functions
# Part I 2) gknockoffX: function generating the knockoff design matrix
# Input:
#   X:      n x p design matrix
#   Omega:  estimated precision matrix (e.g. ISEE estimator)
# Output: 
#   Xnew:   n x (2p) design matrix = [X X_ko]

gknockoffX <- function(X, Omega){
  n = nrow(X)
  p = ncol(X)
  
  # calculate minimum eigenvalue of Sigma = inv(Omega), i.e. 1/max eigenvalue of Omega
  r = eigen(Omega)
  max.eig = r$values[1]
 
  # diagonal matrix for construction of knockoff variables
  s = (1/max.eig)*diag(p)
  obj = sqrtm(2*s - s%*%Omega%*%s)
  B = obj$B
  A = diag(p) - s%*%Omega
  # now construct knockoff variables conditional on X
  X = apply(X, 2, as.numeric);
  X_ko = X%*%A + matrix(rnorm(n*p),n,p)%*%B
  Xnew = cbind(X, X_ko)
  
  return(Xnew)
}
###########################################################################


###########################################################################
# Part I: main functions
# Part I 3) slassotf: function implementing tuning-free sparse regression with scaled Lasso
# Windows version; Mac or Linux version
# Lasso implemented with coordinate descent

slassotf <- function(X, y, regfactor = "log", sis.use = 0, maxite = 50, tol = 1e-2){
  # detect the OS
  if (Sys.info()['sysname'] == 'Windows') {
    windows.use = 1
  }else{
    windows.use = 0}
  
  n = nrow(X)
  p = ncol(X)
  
  if (regfactor == "log") {
    reg.fac = log(n)
  }
  if (regfactor == "one") {
    reg.fac = 1
  }
  if (regfactor == "sqrt") {
    reg.fac = sqrt(n)
  }

  if (sis.use) {
    set0<-order(abs(cor(y, X)),decreasing=T)
    sis.size = min(floor(n/log(n)),p)
    sis.set= set0[1:sis.size]
  }else {
    sis.set = c(1:p)
  }
  
  # rescale each column of n x p design matrix X to have unit L2-norm
  stan_X = scale(X, center=F, scale=T)/sqrt(n-1)
  y.norm = attr(stan_X,"scale")*sqrt(n-1) 
  X = stan_X
  
  # initialization of lambda
  gata2 <- sqrt(n)/(log(p)*2*p)
  B2 <- qt(1-gata2, n-1)
  lambda <- B2/sqrt(n-1+B2^2)
  
  sigma.old <- 1
  beta.old <- c(rep(0, length(sis.set)))
  last_beta_diff <- 0
  inner <- 0
  while (inner < 5) {
    reg <- sigma.old*lambda*reg.fac
    beta <- slasso(as.matrix(X[, sis.set]), as.vector(y), reg, c(rep(0, length(sis.set))), windows.use, maxite, tol)
    if (reg.fac == 1) {
      beta = beta*(abs(beta) > 1*lambda)
    }else{
      beta = beta*(abs(beta) > 0.5*lambda)
    }
      
    sel.mod = which(beta != 0)
    beta[sel.mod] = slasso(as.matrix(X[, sis.set[sel.mod]]), as.vector(y), reg, c(rep(0, length(sel.mod))), windows.use, maxite, tol)
    beta_diff <- max(abs(beta - beta.old))
    A <- y - X[, sis.set] %*% beta
    sigma <- sqrt(sum(A^2))/sqrt(max(n-sum(beta!=0),0)+1e-20)
    sigma_diff <- abs(sigma - sigma.old)
      
    if (sigma_diff < tol) 
      break
    else if (sigma_diff < tol * 0.1& 
              abs(beta_diff - last_beta_diff) < tol) 
      break
    else {
      sigma.old <- sigma
      beta.old <- beta
      last_beta_diff <- beta_diff
    }
    inner <- inner + 1
  }
    
  # beta vector under aligned scale
  betap.stan = c(rep(0, p))
  betap.stan[sis.set] <- as.vector(beta)
  # beta vector under original scale
  betap = scale(t(betap.stan), center=F, scale=y.norm)
  
  obj=list(betap = betap, betap.stan = betap.stan)
  return(obj)
}
###########################################################################


###########################################################################
# Part II: additional functions
# Part II 1) fdr: calculating fdr

fdr <- function(S, beta.true) {
  fdp = sum(beta.true[S] == 0)/max(1, length(S))
  return(fdp)
}
###########################################################################

###########################################################################
# Part II: additional functions
# Part II 2) pow: calculating power

pow <- function(S, beta.true) {
  tdp = sum(beta.true[S] != 0)/sum(beta.true != 0)
  return(tdp)
}
# The end of the master R file
###########################################################################
