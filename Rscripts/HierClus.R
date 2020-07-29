# Hierarchical clustering 
library(cluster)
library(factoextra)
library(fpc)

hclust_setk <- function(sim_matrix, num_clusters){
  hc <- agnes(sim_matrix, diss=TRUE, method="ward")
  pltree(hc, cex = 0.6, hang = -1, main = "Dendrogram")
  nc = num_clusters
  clusterCut <- cutree(hc, nc)
  rect.hclust(tree = hc, k = nc, which = 1:nc, border = 1:nc, cluster = clusterCut)
  return(clusterCut)
}

scls <- hclust_setk(energy_harvesters_smc, 4)
jcls <- hclust_setk(energy_harvesters_jaccard, 4)
ccls <- hclust_setk(energy_harvesters_cosine, 4)
gcls <- hclust_setk(energy_harvesters_ged, 4)
spcls <- hclust_setk(energy_harvesters_spectral, 4)
dcls <- hclust_setk(energy_harvesters_deltacon, 4)
vcls <- hclust_setk(energy_harvesters_veo, 4)
rcls <- hclust_setk(energy_harvesters_resistance, 4)
ncls <- hclust_setk(energy_harvesters_netsimile, 4)

energy_harvesters_vector <- cbind(energy_harvesters_vector, smcCluster = scls )
energy_harvesters_vector <- cbind(energy_harvesters_vector, jacCluster = jcls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, cosCluster = ccls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, gedCluster = gcls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, spectCluster = spcls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, deltaCluster = dcls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, veoCluster = vcls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, resistCluster = rcls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, netsimCluster = ncls)
energy_harvesters_vector <- cbind(energy_harvesters_vector, energy_harvesters_properties)

smc_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$smcCluster), c(338,337, 347,348,349,350,351)]
jaccard_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$jacCluster), c(339,337, 347,348,349,350,351)]
cosine_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$cosCluster), c(340,337, 347,348,349,350,351)]
ged_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$gedCluster), c(341,337, 347,348,349,350,351)]
spectral_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$spectCluster), c(342,337, 347,348,349,350,351)]
deltacon_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$deltaCluster), c(343,337, 347,348,349,350,351)]
veo_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$veoCluster), c(344,337, 347,348,349,350,351)]
resistance_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$resistCluster), c(345,337, 347,348,349,350,351)]
netsimile_grouped <- energy_harvesters_vector[order(energy_harvesters_vector$netsimCluster), c(346,337, 347,348,349,350,351)]

tscls <- hclust_setk(toys_smc, 6)
tjcls <- hclust_setk(toys_jaccard, 5)
tccls <- hclust_setk(toys_cosine, 4)
tgcls <- hclust_setk(toys_ged, 6)
tspcls <- hclust_setk(toys_spectral, 4)
tdcls <- hclust_setk(toys_deltacon, 5)
tvcls <- hclust_setk(toys_veo, 3)
trcls <- hclust_setk(toys_resistance, 4)
tncls <- hclust_setk(toys_netsimile, 5)

toys_vector <- cbind(toys_vector, smcCluster = tscls )
toys_vector <- cbind(toys_vector, jacCluster = tjcls)
toys_vector <- cbind(toys_vector, cosCluster = tccls)
toys_vector <- cbind(toys_vector, gedCluster = tgcls)
toys_vector <- cbind(toys_vector, spectCluster = tspcls)
toys_vector <- cbind(toys_vector, deltaCluster = tdcls)
toys_vector <- cbind(toys_vector, veoCluster = tvcls)
toys_vector <- cbind(toys_vector, resistCluster = trcls)
toys_vector <- cbind(toys_vector, netsimCluster = tncls)
toys_vector <- cbind(toys_vector, toys_properties)

tsmc_grouped <- toys_vector[order(toys_vector$smcCluster), c(112, 121,122,123,124,125)]
tjaccard_grouped <- toys_vector[order(toys_vector$jacCluster), c(113, 121,122,123,124,125)]
tcosine_grouped <- toys_vector[order(toys_vector$cosCluster), c(114, 121,122,123,124,125)]
tged_grouped <- toys_vector[order(toys_vector$gedCluster), c(115, 121,122,123,124,125)]
tspectral_grouped <- toys_vector[order(toys_vector$spectCluster), c(116, 121,122,123,124,125)]
tdeltacon_grouped <- toys_vector[order(toys_vector$deltaCluster), c(117, 121,122,123,124,125)]
tveo_grouped <- toys_vector[order(toys_vector$veoCluster), c(118, 121,122,123,124,125)]
tresistance_grouped <- toys_vector[order(toys_vector$resistCluster), c(119, 121,122,123,124,125)]
tnetsimile_grouped <- toys_vector[order(toys_vector$netsimCluster), c(120, 121,122,123,124,125)]

## plotting all sorts of property histograms
ggplot(deltacon_grouped, aes(x=complexity)) + geom_histogram(colour="black", fill="white") + 
  facet_grid(deltaCluster ~ .)

## getting a list of nodes
test <- unlist(lapply(list(strsplit(deltacon_grouped[deltacon_grouped$deltaCluster == 1,]$max_deg_nodes, ",")) , function(i) list(unlist(i, recursive = TRUE))))
barplot(table(test))

## getting a bar plot of categories
melted <- melt(table(deltacon_grouped[c(1,2)]), id.vars = c("deltaCluster"))
ggplot(melted, aes(fill=category, y=value, x=deltaCluster)) + 
     geom_bar(position="fill", stat="identity")
