
setwd("/home/yanglu/Downloads/RANK/")
source('/home/yanglu/Downloads/RANK/gknockoff_all.R')
source('/home/yanglu/Downloads/RANK/isee_all.R')

# load packages
require(graphics)
library(igraph)
library(MASS) 
library(Matrix)
library(pracma)

regfactor = "log"
npermu = 5
sis.use = 0
bia.cor = 0

xURL = "/home/yanglu/Downloads/deepknock/out_wbigangp_Sphr1Gaus1_BNTrue_z128_p200/x_itr500000.csv"
xKnockURL = "/home/yanglu/Downloads/deepknock/out_wbigangp_Sphr1Gaus1_BNTrue_z128_p200/x_itr500000_knockoff.csv"
zURL = "/home/yanglu/Downloads/deepknock/out_wbigangp_Sphr1Gaus1_BNTrue_z128_p200/z_itr500000.csv"
zKnockURL = "/home/yanglu/Downloads/deepknock/out_wbigangp_Sphr1Gaus1_BNTrue_z128_p200/z_itr500000_knockoff.csv"

#X <- read.csv(xURL, header = FALSE, sep = ",")
#n <- nrow(X); print(n)
#p <- ncol(X); print(p)

#obj = isee(X, regfactor, npermu, sis.use, bia.cor) 
#if (bia.cor == 1){
#  Omega = obj$Omega.isee.c
#} else {
#  Omega = obj$Omega.isee
#}

#Xnew = gknockoffX(X, Omega)
#print(nrow(Xnew))
#print(ncol(Xnew))
#write.table(Xnew, file=xKnockURL,row.names=FALSE, col.names=FALSE, sep=",")

#rm(X,obj,Omega,Xnew)


Z <- read.csv(zURL, header = FALSE, sep = ",")

obj = isee(Z, regfactor, npermu, sis.use, bia.cor) 
if (bia.cor == 1){
  Omega = obj$Omega.isee.c
} else {
  Omega = obj$Omega.isee
}

Znew = gknockoffX(Z, Omega)
print(nrow(Znew))
print(ncol(Znew))
write.table(Znew, file=zKnockURL,row.names=FALSE, col.names=FALSE, sep=",")

rm(Z,obj,Omega,Znew)


