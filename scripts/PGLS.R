library(ape)
library(geiger)
library(nlme)
library(phytools)

setwd("~/Documents/SelCoeff")

mamData <- read.csv("experiments/3bins-mC-OntoRegTrans/regression-MutSel/results.tsv", row.names = 1)
mamTree <- read.tree("data_processed/trimmed_tree_sample_all.tsv.tree")
plot(mamTree)

name.check(mamTree, mamData)

y_label <- "mut_sum_P.Sweak"
y_label <- "bayes_P.Spos_P.pos"
plot(mamData[, c("pop_size", y_label)])

# Extract columns
x <- mamData[, "pop_size"]
y <- mamData[, y_label]
# Give them names
names(y) <- names(x) <- rownames(mamData)

# pgls model
pglsModel <- gls(y ~ x, correlation = corBrownian(phy = mamTree),
                 data = mamData, method = "ML")
summary(pglsModel)
t <- summary(pglsModel)$tTable

# Calculate PICs
yPic <- pic(y, mamTree)
xPic <- pic(x, mamTree)
# Make a model
picModel <- lm(yPic ~ xPic - 1)
# Yes, significant
summary(picModel)

# plot yPic
plot(yPic ~ xPic)
abline(a = 0, b = coef(picModel))

