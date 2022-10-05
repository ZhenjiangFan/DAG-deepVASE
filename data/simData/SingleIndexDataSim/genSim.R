#
# This is an example of applying the graphical knockoff filter.
#
# The master file "isee_all.R" includes all the R functions for implementing the innovated 
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


# detect the OS & set working directory
# detect the OS & set working directory

setwd("/home/yanglu/Downloads/RANK/")
source('/home/yanglu/Downloads/RANK/gknockoff_all.R')
source('/home/yanglu/Downloads/RANK/isee_all.R')

# load packages
require(graphics)
library(igraph)
library(MASS) 
library(Matrix)
library(pracma)

# simulation parameters
regfactor = "log"
npermu = 5
sis.use = 0
bia.cor = 0

n = 2000
n1 = 400
p = 200
sig = 0.5
obj = band.mat(a=0.5, p, K = 1)
Sig.half = obj$Sigma.half
Ome.true = obj$Omega
#q = 0.2
#X = make.data(Sig.half, n, p, seed = 1000)

beta.true = c(rep(0, p))
S.true = c(1:30)
beta.true[S.true] = c(1,-1,1,-1,1,-1,1,1,-1,1,1,-1,1,-1,1,-1,1,1,-1,1,1,-1,1,-1,1,-1,1,1,-1,1)

qList = c(0.2,0.1,0.05)
iterAll = 5;

for(q in qList) {
print(paste("q=", toString(q)))

fdp1  = rep(0,iterAll)
tdp1  = rep(0,iterAll)
fdp2  = rep(0,iterAll)
tdp2  = rep(0,iterAll)


for (iter in 1:iterAll)
{
        xURL = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/X_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        yURL1 = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/y_lr_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        yURL2 = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/y_plr_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        yURL3 = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/y_si_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        
        X_sub <- data.matrix(read.csv(xURL, header = FALSE, sep = ","))

        #X_sub = X[((iter-1)*n1+1):(iter*n1),]
        #write.table(X_sub, file=paste("../deepknock/X_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=FALSE, sep=",")
        print(paste("X_sub row: ",toString(nrow(X_sub)),"  col:",toString(ncol(X_sub))))
        
        #eps = sig*matrix(rnorm(n1),n1,1)
        #gu  = matrix(sin(runif(n1)*2*pi),n1,1)
        #y1 = X_sub%*%beta.true + eps
        #y2 = X_sub%*%beta.true + gu + eps
        #y3 = 0.5*((X_sub%*%beta.true)^3) + eps

        #write.table(y1, file=paste("../deepknock/y_lr_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=FALSE, sep=",")
        #write.table(y2, file=paste("../deepknock/y_plr_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=FALSE, sep=",")
        #write.table(y3, file=paste("../deepknock/y_si_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=FALSE, sep=",")
        
        y1 <- data.matrix(read.csv(yURL1, header = FALSE, sep = ","))
        #print(paste("y1 row: ",toString(nrow(y1)),"  col:",toString(ncol(y1))))
        y2 <- data.matrix(read.csv(yURL2, header = FALSE, sep = ","))
        #print(paste("y2 row: ",toString(nrow(y2)),"  col:",toString(ncol(y2))))

        S1 = gknockoff(X_sub, y1, q, regfactor = regfactor); 
        fdp1[iter] = fdr(S1, beta.true);
        tdp1[iter] = pow(S1, beta.true);
        #print(fdp1); rm(S1);
        
        S2 = gknockoff(X_sub, y2, q, regfactor = regfactor); 
        fdp2[iter] = fdr(S2, beta.true);
        tdp2[iter] = pow(S2, beta.true);
        #print(fdp2); rm(S2);
        
        rm(X_sub,y1,y2)
        
        #obj = isee(X_sub, regfactor, npermu, sis.use, bia.cor) 
        #Omega = obj$Omega.isee

        #Xnew = data.matrix(gknockoffX(X_sub, Omega))
        #print(paste("X_new row: ",toString(nrow(Xnew)),"  col:",toString(ncol(Xnew))))
        #print(Xnew)
        
        #obj1 = slassotf(Xnew, y1, regfactor, sis.use = 0)
        #betap.stan = obj1$betap.stan

        #W = c(rep(0, p))
        #for (j in 1:p) {
        #  W[j] = abs(betap.stan[j]) - abs(betap.stan[j+p])
        #}

        #t = sort(c(0, abs(W)))
        #ratio = c(rep(0, p))
        #for (j in 1:p) {
        #  ratio[j] = sum(W <= -t[j])/max(1, sum(W >= t[j]))
        #}
        #id = which(ratio <= q)[1]
        #if(length(id) == 0){
        #  T = Inf
        #} else {
        #  T = t[id]
        #}


        #S = which(W >= T); print(S)
        #fdp = fdr(S, beta.true); print(fdp)
        #tdp = pow(S, beta.true); print(tdp)
        #rm(obj,Omega,Xnew,betap.stan,W,t,ratio,id,S,fdp,tdp)
        #rm(X_sub,eps,gu,y1,y2,y3)
}

print(paste("fdp1: ", toString(mean(fdp1))," , ", toString(sd(fdp1))))
print(paste("tdp1: ", toString(mean(tdp1))," , ", toString(sd(tdp1))))
print(paste("fdp2: ", toString(mean(fdp2))," , ", toString(sd(fdp2))))
print(paste("tdp2: ", toString(mean(tdp2))," , ", toString(sd(tdp2))))
}



beta.true = c(rep(0, p))
S.true = c(1:10)
beta.true[S.true] = c(1,-1,1,-1,1,-1,1,1,-1,1)

qList = c(0.2,0.1,0.05)
iterAll = 5;

for(q in qList) {
print(paste("q=", toString(q)))

fdp3  = rep(0,iterAll)
tdp3  = rep(0,iterAll)


for (iter in 1:iterAll)
{
        xURL = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/X_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        yURL1 = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/y_lr_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        yURL2 = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/y_plr_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        yURL3 = paste("~/Documents/deepknockoff/data/2017-11-18/n",toString(n1),"_p",toString(p),"/y_si_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep="")
        
        X_sub <- data.matrix(read.csv(xURL, header = FALSE, sep = ","))

        #X_sub = X[((iter-1)*n1+1):(iter*n1),]
        #write.table(X_sub, file=paste("../deepknock/X_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=FALSE, sep=",")
        print(paste("X_sub row: ",toString(nrow(X_sub)),"  col:",toString(ncol(X_sub))))
        
        #eps = sig*matrix(rnorm(n1),n1,1)
        #gu  = matrix(sin(runif(n1)*2*pi),n1,1)
        #y1 = X_sub%*%beta.true + eps
        #y2 = X_sub%*%beta.true + gu + eps
        #y3 = 0.5*((X_sub%*%beta.true)^3) + eps

        #write.table(y3, file=paste("../deepknock/y_si_n",toString(n1),"_p",toString(p),"_iter",toString(iter),".csv",sep=""),row.names=FALSE, col.names=FALSE, sep=",")
        
        y3 <- data.matrix(read.csv(yURL3, header = FALSE, sep = ","))
        #print(paste("y3 row: ",toString(nrow(y3)),"  col:",toString(ncol(y3))))
        
        
        S3 = gknockoff(X_sub, y3, q, regfactor = regfactor); 
        fdp3[iter] = fdr(S3, beta.true);
        tdp3[iter] = pow(S3, beta.true);
        #print(fdp3); rm(S3);
        
        rm(X_sub,y3)
        
        
        #obj = isee(X_sub, regfactor, npermu, sis.use, bia.cor) 
        #Omega = obj$Omega.isee

        #Xnew = data.matrix(gknockoffX(X_sub, Omega))
        #print(paste("X_new row: ",toString(nrow(Xnew)),"  col:",toString(ncol(Xnew))))
        #print(Xnew)
        
        #obj1 = slassotf(Xnew, y1, regfactor, sis.use = 0)
        #betap.stan = obj1$betap.stan

        #W = c(rep(0, p))
        #for (j in 1:p) {
        #  W[j] = abs(betap.stan[j]) - abs(betap.stan[j+p])
        #}

        #t = sort(c(0, abs(W)))
        #ratio = c(rep(0, p))
        #for (j in 1:p) {
        #  ratio[j] = sum(W <= -t[j])/max(1, sum(W >= t[j]))
        #}
        #id = which(ratio <= q)[1]
        #if(length(id) == 0){
        #  T = Inf
        #} else {
        #  T = t[id]
        #}


        #S = which(W >= T); print(S)
        #fdp = fdr(S, beta.true); print(fdp)
        #tdp = pow(S, beta.true); print(tdp)
        #rm(obj,Omega,Xnew,betap.stan,W,t,ratio,id,S,fdp,tdp)
        #rm(X_sub,eps,gu,y1,y2,y3)
}

print(paste("fdp3: ", toString(mean(fdp3))," , ", toString(sd(fdp3))))
print(paste("tdp3: ", toString(mean(tdp3))," , ", toString(sd(tdp3))))

}

