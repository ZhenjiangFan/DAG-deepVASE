setwd("C:\\Users\\zhenjiang fan\\Documents\\workspace\\Pythonworkspace\\Feature selection using deep learning\\data simulation\\RANK")
source('isee_all.R')


# load packages
library(MASS) 
library(Matrix)


p = 40
n=2000
iterAll = 20;
dataDir = paste("/ihome/hpark/zhf16/causalDeepVASE/data/simulated_data/",toString(n),"samples/",toString(p),"p/",sep="")

sig = 0.5
beta.true = c(rep(0, p))
S.true = c(1:20)
#beta.true[S.true] = c(1,-1,1,-1,1,-1,1,1,-1,1)
beta.true[S.true] = c(1,1,1,1,1,1,1,1,1,1
                      ,1,1,1,1,1,1,1,1,1,1)

#q = 0.2
effectValue = 2;#0.5
alphaValue = 1#,0.5,0.7

for (iter in 1:iterAll){
  obj = band.mat(a=0.5, p, K = 1)
  Sig.half = obj$Sigma.half
  Ome.true = obj$Omega
  X = make.data(Sig.half, n, p, seed = 1000)
  
  print(paste("Iter: ",toString(iter)))
  print(paste("X row: ",toString(nrow(X)),"  col:",toString(ncol(X))))
  write.table(X, file=paste(dataDir,"X_n",toString(n),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=TRUE, sep=",")
  eps = sig*matrix(rnorm(n),n,1)
  #Y = effectValue*((X%*%beta.true)^3) + eps
  #Y = effectValue*alphaValue*((X%*%beta.true)^3) + (1-alphaValue)*X%*%beta.true + eps
  Y = effectValue*alphaValue*((X%*%beta.true)^2) + (1-alphaValue)*X%*%beta.true + eps
  write.table(Y, file=paste(dataDir,"y_si_n",toString(n),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=TRUE, sep=",")
}

