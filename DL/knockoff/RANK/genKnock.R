#Install R environment if using conda: #conda install -c r r
#Load the R files. Use "\\" on Windows and "/" on Linux
gknockoff_all_path = paste(normalizePath(dirname("gknockoff_all.R")),"/DL/knockoff/RANK/", "gknockoff_all.R", sep = "")
source(gknockoff_all_path)
#source('/ihome/hpark/zhf16/test/DeepPINK/knockoff/RANK/gknockoff_all.R')

isee_all_path = paste(normalizePath(dirname("isee_all.R")),"/DL/knockoff/RANK/", "isee_all.R", sep = "")
# source('/ihome/hpark/zhf16/test/DeepPINK/knockoff/RANK/isee_all.R')
source(isee_all_path)

# load packages
require(graphics)
library(igraph)
library(MASS) 
library(Matrix)
library(pracma)

generateKnockoff <- function(data_dir, data_name, regfactor = "log", npermu = 5, sis.use = 0, bia.cor = 0){
    #regfactor = "log"
    #npermu = 5
    #sis.use = 0
    #bia.cor = 0
    #print(regfactor)
    #print(npermu)
    #print(sis.use)
    xURL <- paste(data_dir, data_name, sep ="/")
    data_name2 = gsub("\\.csv", "", data_name)
    knockoff_name = paste(data_name2,"_knockoff.csv" , sep ="")
    
    xKnockURL <- paste(data_dir, knockoff_name, sep ="/")
    
    X <- read.csv(xURL, header = FALSE, sep = ",")
    n <- nrow(X); print(n)
    p <- ncol(X); print(p)
  
    obj = isee(X, regfactor, npermu, sis.use, bia.cor) 
    if (bia.cor == 1){
        Omega = obj$Omega.isee.c
    } else {
        Omega = obj$Omega.isee
    }
  
    Xnew = gknockoffX(X, Omega)
    #print(nrow(Xnew))
    #print(ncol(Xnew))
    write.table(Xnew, file=xKnockURL,row.names=FALSE, col.names=FALSE, sep=",")
    #print(xKnockURL)
    rm(X,obj,Omega,Xnew)
    
    return(xKnockURL)
  
}
