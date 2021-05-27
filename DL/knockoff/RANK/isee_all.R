#
# This master R file includes all the R functions for implementing the innovated 
# scalable efficient estimation (ISEE) procedure introduced in Fan and Lv (2015) for 
# scalable estimation of both the innovated data matrix (n x p matrix X tilde = X Omega) 
# and the precision matrix (Omega = inverse covariance matrix) in ultra-large scales.
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
#   1) isee: main function calculating hat matrix for Omega
#   2) isee.X: function calculating hat matrix for X tilde
#   3) slasso: function implementing tuning-free sparse regression with scaled Lasso
# Part II: additional functions
#   1) beta.block: function calculating beta hat matrix for each small block
#   2) isee.cv: cv function selecting threshold tau
# Part III: functions summarizing results and generating data matrix
#   1) rate.func: function summarizing selection results
#   2) make.data: function generating data matrix
#   3) blk.mat: function generating block diagonal Omega
#   4) ar1.mat: function generating AR(1) Omega
#   5) band.mat: function generating band Omega
#

###########################################################################
# Part I: main functions
# Part I 1) isee: main function calculating hat matrix for Omega

isee <- function(X, regfactor = "log", npermu = 5, sis.use = 0, bia.cor = 0){
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
  
  Omega.isee = matrix(0, p, p)
  # bias corrected ISEE estimator
  Omega.isee.c = matrix(0, p, p)
  temp = matrix(0, p, p)
  temp1 = matrix(0, p, p)
  for (i in 1:npermu) {
    if (i == 1) {
      permu = c(1:p)
    } else {
      permu = sample.int(p,size=p)
    }
    
    obj = isee.X(X, permu, reg.fac, sis.use, bia.cor)
    X.tilde = obj$X.tilde
    
    obj1 = isee.cv(X.tilde)
    Omega.isee = Omega.isee + obj1$Omega.isee
    temp = temp + (obj1$Omega.isee != 0)
    
    if (bia.cor == 1){
      # bias corrected ISEE estimator with correction on its support
      Omega.isee.c0 = obj1$Omega.isee + (obj1$Omega.isee != 0)*(abs(cov2cor(obj$biascor.mat)) > obj1$tau)*(obj$biascor.mat - obj1$Omega.isee)/2
      Omega.isee.c = Omega.isee.c + Omega.isee.c0
      temp1 = temp1 + (Omega.isee.c0 != 0)
    }
  }
  
  Omega.isee = Omega.isee/(temp + matrix(1e-8, p, p))
  if (bia.cor == 1){
    Omega.isee.c = (Omega.isee != 0)*Omega.isee.c/(temp1 + matrix(1e-8, p, p))
  }
  
  if (bia.cor == 0){
    obj=list(Omega.isee=Omega.isee)
  }
  if (bia.cor == 1){
    obj=list(Omega.isee=Omega.isee, Omega.isee.c=Omega.isee.c)
  }
  return(obj)
}
###########################################################################

# Part I 2) isee.X: function calculating hat matrix for X tilde

isee.X <- function(X, permu, reg.fac, sis.use = 0, bia.cor = 0){
  # detect the OS
  if (Sys.info()['sysname'] == 'Windows') {
    windows.use = 1
  }else{
    windows.use = 0}
  
  X = X[,permu] 
  # K is the blocksize
  K <-2
  p <- ncol(X)
  n <- nrow(X)
  # bias corrected matrix
  biascor.mat <- matrix(0, p, p)
  # beta coefficient matrix
  beta.mat <- matrix(0, p, p)
  
  gata2 <- sqrt(n)/(log(p)*2*p)
  B2 <- qt(1-gata2, n-1)
  lambda <- B2/sqrt(n-1+B2^2)
  
  stan_X = scale(X, center=F, scale=T)/sqrt(n-1)
  y.norm = attr(stan_X,"scale")*sqrt(n-1) 
  i.index <- seq(1,p-1,by=K)
  
  X.tilde = matrix(0, nrow=n, ncol=p)
  for (k in 1:length(i.index)){ 
    i = i.index[k]
    
    if(i==p-2){
      j=c(p-1,p)
    }else{
      j=i+1
    }
    
    block.ind = c(i,j)
    temp= beta.block(stan_X, block.ind, lambda, reg.fac, windows.use, sis.use)
    beta.coef = temp$beta.j
    
    temp2 = scale(stan_X[,-block.ind] %*% beta.coef, center=F, scale=1/y.norm[block.ind])
    epsi <-  apply(X[, block.ind] - temp2, 2, as.numeric)
    omega <- solve(t(epsi)%*%epsi/n)
    
    
    X.tilde[,block.ind] =  as.matrix(epsi%*%omega)
    
    if (bia.cor == 1){
      biascor.mat[block.ind,block.ind] = as.matrix(omega)
      beta.mat[block.ind,-block.ind] = t(scale(beta.coef, center=F, scale=1/y.norm[block.ind]))
    }
  }

  if (bia.cor == 1){
    beta.mat = scale(beta.mat, center=F, scale=y.norm)
    Omega.ini = t(X.tilde)%*%X.tilde/n
    
    for (ii in 1:(length(i.index)-1)){
      for (jj in (ii+1):length(i.index)){
        i = i.index[ii]
        indset1 = i:(i+K-1)
        j = i.index[jj]
        indset2 = j:(j+K-1)
        
        biascor.mat[indset1,indset2] = -(Omega.ini[indset1,indset2] + t(beta.mat[indset1,indset2])%*%Omega.ini[indset1,indset1] + t(beta.mat[indset2,indset1])%*%Omega.ini[indset2,indset2])
        biascor.mat[indset2,indset1] = t(biascor.mat[indset1,indset2])
      }
    }
    biascor.mat = biascor.mat[order(permu),order(permu)]
  }
  
  X.tilde = X.tilde[,order(permu)]
  if (bia.cor == 0){
    obj=list(X.tilde=X.tilde)
  }
  if (bia.cor == 1){
    obj=list(X.tilde=X.tilde, biascor.mat=biascor.mat)
  }
  return(obj)
}
###########################################################################

# Part I 3) slasso: function implementing tuning-free sparse regression with scaled Lasso

# Windows version; Mac or Linux version
# Lasso implemented with coordinate descent
# each column of n x p design matrix X is rescaled to have unit L2-norm

slasso <- function(X, y, lam, betap, windows.use = 1, maxite = 50, tol = 1e-2){
  ## detect the OS
  #if (Sys.info()['sysname'] == 'Windows') {
  #  windows.use = 1
  #}else{
  #  windows.use = 0}
  
  nr = nrow(X)
  nc = ncol(X)
  if (windows.use) {
    dyn.load("DL/knockoff/RANK/slasso.dll")
  } else {
    dyn.load("DL/knockoff/RANK/slasso.so")
  }
  # for output variables, use outvar = some argument
  out = .C("slasso", as.vector(as.numeric(X)), as.integer(nr), as.integer(nc), as.double(y), as.double(lam), as.integer(maxite), as.double(tol), betap = as.double(betap))
  # take out betap component of the returned list
  if (is.nan(out$betap[1])) stop("NaN values")
  return(out$betap)
}
###########################################################################


###########################################################################
# Part II: additional functions
# Part II 1) beta.block: function calculating beta hat matrix for each small block

beta.block <- function(X, block.ind, lambda, reg.fac, windows.use, sis.use){
  p <- ncol(X)
  n <- nrow(X)
  tol = 1e-3
  block.size = length(block.ind)
  precision <- matrix(0, ncol = block.size , nrow = block.size)
  
  beta.j<- matrix(0, nrow = p-block.size, ncol = block.size)
  
  for (j in 1:block.size){
    if (sis.use) {
      org.set = c(1:p)[ -block.ind]
      set0<-order(abs(cor(X[, -block.ind], X[, block.ind[j]])),decreasing=T)
      sis.size = min(floor(n/log(n)),p-block.size)
      sis.set0<-sort(set0[1:sis.size])
      sis.set= org.set[sis.set0]
    }else {
      sis.set = c(1:p)[ -block.ind]
      sis.set0 = c(1:length(sis.set)) 
    }
    
    sigma.old <- 1
    beta.old <- c(rep(0, length(sis.set)))
    last_beta_diff <- 0
    inner <- 0
    while (inner < 5) {
      reg <- sigma.old*lambda*reg.fac
      beta <- slasso(as.matrix(X[, sis.set]), as.matrix(X[, block.ind[j]]),reg, c(rep(0, length(sis.set))), windows.use)
      if (reg.fac == 1) {
        beta = beta*(abs(beta) > 1*lambda)
      }else{
        beta = beta*(abs(beta) > 0.5*lambda)
      }
      
      sel.mod = which(beta != 0)
      beta[sel.mod] = slasso(as.matrix(X[, sis.set[sel.mod]]), as.matrix(X[, block.ind[j]]), reg, c(rep(0, length(sel.mod))), windows.use)
      beta_diff <- max(abs(beta - beta.old))
      A <- X[, block.ind[j]] - X[, sis.set] %*% beta
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
    beta.j[sis.set0,j] <- as.vector(beta)
  }
  obj=list(beta.j=beta.j)
  return(obj)
}
###########################################################################

# Part II 2) isee.cv: cv function selecting threshold tau

isee.cv <- function(X.tilde, ntau = 20, split.ratio = 0.9, n.split=5, criterion="Frob"){
  n=nrow(X.tilde)
  p=ncol(X.tilde)
  n.train = ceiling(n*split.ratio)
  n.test = n-n.train
  
  tau.min = 0.5
  tau.max = 1
  tau.path = seq(tau.min,tau.max,length.out=ntau)*sqrt(log(p)/n)
  
  # loss function used in calculating prediction error
  loss.func <- function(Sig, Sig.tau, criterion){
    if(criterion=="Frob")err= sum((Sig-Sig.tau)^2)
    if(criterion=="Infi")err= max(abs(Sig-Sig.tau))
    if(criterion=="L1")err = sum(abs(Sig-Sig.tau))
    if(criterion=="L1L1")err = sum(abs(Sig-Sig.tau))+0.5*sum(abs(Sig.tau))
    return(err)
  }
  
  err = matrix(0, n.split, length(tau.path))
  for(j in 1:n.split){
    train.id = sample.int(n, size = n.train, replace = FALSE)
    test.id = setdiff(c(1:n), train.id)
    Sig.train = cor(X.tilde[train.id,])
    Sig.test = cor(X.tilde[test.id,])
    
    for(i in 1:length(tau.path)){
      Sig.train.tau = Sig.train*(abs(Sig.train)>tau.path[i])
      err[j,i] = loss.func(Sig.test,Sig.train.tau, criterion)
    }
  }
  
  pe.vec = apply(err, 2, mean)
  tau.id = which.min(pe.vec)
  
  Omega.ini = (t(X.tilde)%*%X.tilde/n)
  Omega.isee = Omega.ini*(abs(cov2cor(Omega.ini)) > tau.path[tau.id])
  
  obj=list(Omega.isee=Omega.isee, tau=tau.path[tau.id])
  return(obj)
}
###########################################################################


###########################################################################
# Part III: functions summarizing results and generating data matrix
# Part III 1) rate.func: function summarizing selection results

rate.func <- function(Omega.hat, Omega){
  TP <- sum((Omega.hat!=0)*(Omega!=0))
  FP <- sum((Omega.hat!=0)*(Omega==0))
  TN <- sum((Omega.hat == 0)*(Omega==0))
  FN <- sum((Omega.hat == 0)*(Omega!=0))
  
  TPR <- TP/(TP+FN)
  FPR <- FP/(FP+TN) 
  obj <-list(TPR=TPR, FPR=FPR)
  return(obj)
}
###########################################################################

# Part III 2) make.data: function generating data matrix

make.data <- function(Sigma.half, n, p, seed){
  set.seed(seed)  
  
  X = matrix(rnorm(n*p),n,p)%*%Sigma.half
  return(X)
}
###########################################################################
###########################################################################

# Part III 2-1) make.data: function generating data matrix

make.data1 <- function(Sigma.half, n, p){
##  set.seed(seed)  
  
  X = matrix(rnorm(n*p),n,p)%*%Sigma.half
  return(X)
}
###########################################################################

# Part III 3) blk.mat: function generating block diagonal Omega

blk.mat <- function(a=0.5, p, permu){
  Omega.blk = matrix(c(1,a,a,1),2,2)
  Sigma.blk = solve(Omega.blk)
  Sigma.half.blk =  chol(Sigma.blk)
  Omega = Omega.blk
  Sigma = Sigma.blk
  Sigma.half = Sigma.half.blk
  
  for(j in 1:((p/2)-1)){
    Omega=bdiag(Omega, Omega.blk)
    Sigma = bdiag(Sigma, Sigma.blk)
    Sigma.half = bdiag(Sigma.half, Sigma.half.blk)
  }
  
  Sigma = Sigma[permu,permu]
  Omega = Omega[permu,permu]
  Sigma.half = Sigma.half[permu,permu]
  data = list(Sigma = Sigma, Omega=Omega, Sigma.half = Sigma.half)
  return(data)
}
###########################################################################

# Part III 4) ar1.mat: function generating AR(1) Omega

ar1.mat <- function(a, p, permu=c(1:p)){
  times <- 1:p
  H <- abs(outer(times, times, "-"))
  Omega<- a^H
  Sigma = qr.solve(Omega) 
  Sigma = Sigma*(abs(Sigma)>1e-4)
  Sigma.half = chol(Sigma)
  Sigma.half = Sigma.half*(abs(Sigma.half)>1e-4)
  Sigma = Sigma[permu,permu]
  Omega = Omega[permu,permu]
  Sigma.half = Sigma.half[permu,permu]
  obj = list(Sigma=Sigma, Omega = Omega, Sigma.half = Sigma.half)
}
###########################################################################

# Part III 5) band.mat: function generating band Omega

band.mat <- function(a, p, K=1, permu=c(1:p)){
  ones = rep(1,p)
  Omega0 = a*ones%*%t(ones)
  diag(Omega0) = rep(1,p)
  Omega = 1*band(Omega0,-K,K)
  Sigma = qr.solve(Omega)
  Sigma = Sigma*(abs(Sigma)>1e-4)
  Sigma.half=chol(Sigma)
  Sigma.half = Sigma.half*(abs(Sigma.half)>1e-4)
  Sigma = Sigma[permu,permu]
  Omega = Omega[permu,permu]
  Sigma.half = Sigma.half[permu,permu]
  obj = list(Sigma=Sigma, Omega = Omega, Sigma.half = Sigma.half)
}
# The end of the master R file
###########################################################################
