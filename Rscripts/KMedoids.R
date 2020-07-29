setwd("C:/Users/nandy/source/repos/FunctionalModelSimilarity/FunctionalModelSimilarity/Rscripts") 
energy_harvesters_deltacon <- read.csv("../data/03_processed/energy_harvesters_deltacon_sorted.csv", row.names=1)
energy_harvesters_ged <- read.csv("../data/03_processed/energy_harvesters_geddistance_sorted.csv", row.names=1)
energy_harvesters_spectral <- read.csv("../data/03_processed/energy_harvesters_lambdadistance_sorted.csv", row.names=1)
energy_harvesters_smc <- read.csv("../data/03_processed/energy_harvesters_hammingdist.csv", row.names=1)
energy_harvesters_jaccard <- read.csv("../data/03_processed/energy_harvesters_jaccarddist.csv", row.names=1)
energy_harvesters_cosine <- read.csv("../data/03_processed/energy_harvesters_cosinedist.csv", row.names=1)
energy_harvesters_veo <- read.csv("../data/03_processed/energy_harvesters_vertexedgeoverlap.csv", row.names=1)
energy_harvesters_netsimile <- read.csv("../data/03_processed/energy_harvesters_netsimile.csv", row.names=1)
energy_harvesters_resistance <- read.csv("../data/03_processed/energy_harvesters_resistance.csv", row.names=1)
energy_harvesters_vector <- read.csv("../data/02_intermediate/energy_harvesters_vector.csv", row.names=1)
energy_harvesters_properties <- read.csv("../data/03_processed/energy_harvesters_properties.csv", row.names=1)

toys_deltacon <- read.csv("../data/03_processed/appliances_and_toys_deltacon_V2.csv", row.names=1)
toys_ged <- read.csv("../data/03_processed/appliances_and_toys_geddistance_V2.csv", row.names=1)
toys_spectral <- read.csv("../data/03_processed/appliances_and_toys_lambdadistance_V2.csv", row.names=1)
toys_smc <- read.csv("../data/03_processed/appliances_and_toys_hammingdist_V2.csv", row.names=1)
toys_jaccard <- read.csv("../data/03_processed/appliances_and_toys_jaccarddist_V2.csv", row.names=1)
toys_cosine <- read.csv("../data/03_processed/appliances_and_toys_cosinedist_V2.csv", row.names=1)
toys_veo <- read.csv("../data/03_processed/appliances_and_toys_vertexedgeoverlap_V2.csv", row.names=1)
toys_netsimile <- read.csv("../data/03_processed/appliances_and_toys_netsimile_V2.csv", row.names=1)
toys_resistance <- read.csv("../data/03_processed/appliances_and_toys_resistance_V2.csv", row.names=1)
toys_vector <- read.csv("../data/02_intermediate/appliances_and_toys_vector_V2.csv", row.names=1)
toys_properties <- read.csv("../data/03_processed/appliances_and_toys_V2_properties.csv", row.names=1)

energy_harvesters_vector <- energy_harvesters_vector[-c(1, 2), ]
toys_vector <- toys_vector[-c(1), ]
#energy_harvesters_vector <- cbind(energy_harvesters_vector, complexity = rowSums(sapply(energy_harvesters_vector, as.numeric)))

harvestertype <- c(rep(('inductive'),each=9),
rep(('piezoelectric'),each=6),
rep(('wind'),each=6),
rep(('wave'),each=3), 
rep(('solar'),each=6),
rep(('thermal'),each=5),
rep(('hybrid'),each=4))

energy_harvesters_vector <- cbind(energy_harvesters_vector, category = harvestertype)

library(factoextra)
library(cluster)

kmed_setk <- function(smc, jac, cos, ged, lambda, dc, k){
  pam.res1 <- pam(scale(smc), k)
  pam.res2 <- pam(scale(jac), k)
  pam.res3 <- pam(scale(cos), k)
  pam.res4 <- pam(scale(ged), k)
  pam.res5 <- pam(scale(lambda), k)
  pam.res6 <- pam(scale(dc), k)
  pam.results <- list("smc" = pam.res1, "jaccard" = pam.res2, "cosine" = pam.res3, 
                      "ged" = pam.res4, "spectral" = pam.res5, "deltacon" = pam.res6)
  return(pam.results)
}

cls <- kmed_setk(energy_harvesters_smc, energy_harvesters_jaccard, energy_harvesters_cosine, 
          energy_harvesters_ged, energy_harvesters_spectral, energy_harvesters_deltacon, 2)
cls <- kmed_setk(toys_smc, toys_jaccard, toys_cosine, 
                 toys_ged, toys_spectral, toys_deltacon, 2)
fviz_cluster(cls$smc)
fviz_cluster(cls$jaccard)
fviz_cluster(cls$cosine)
fviz_cluster(cls$ged)
fviz_cluster(cls$spectral)
fviz_cluster(cls$deltacon)

energy_harvesters_vector <- cbind(energy_harvesters_vector, cls_smc = cls$smc$cluster)
energy_harvesters_vector <- cbind(energy_harvesters_vector, cls_jaccard = cls$jaccard$cluster)
energy_harvesters_vector <- cbind(energy_harvesters_vector, cls_cosine = cls$cosine$cluster)
energy_harvesters_vector <- cbind(energy_harvesters_vector, cls_ged = cls$ged$cluster)
energy_harvesters_vector <- cbind(energy_harvesters_vector, cls_spectral = cls$spectral$cluster)
energy_harvesters_vector <- cbind(energy_harvesters_vector, cls_deltacon = cls$deltacon$cluster)

library(fpc)
pamk(toys_deltacon,krange=1:60,criterion="asw", usepam=TRUE,
     scaling=TRUE, alpha=0.001, diss=TRUE, critout=FALSE)


