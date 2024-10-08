library("argparse")
library("geiger")
library("caper")

# create parser object
parser <- ArgumentParser()
parser$add_argument("--input_tsv", help = "Input tsv file")
parser$add_argument("--input_tree", help = "Input tree file")
parser$add_argument("--output_tsv", help = "Output tsv file")

args <- parser$parse_args()
mamData <- read.csv(args$input_tsv, row.names = 1, sep = "\t", check.names = FALSE)
mamData$Names <- rownames(mamData)

mamTree <- read.tree(args$input_tree)
name.check(mamTree, mamData)

table_output <- data.frame()
y_list <- colnames(mamData)
print(y_list)
for (x_label in c("pop_size", "all_dS", "all_P-Spos", "all_P-Sneg", "all_P-Sweak")) {
    for (y_label in y_list) {
        if ((y_label == x_label) || (y_label == "Names") || (y_label == "species")) {
            next
        }
        # Extract columns
        print(x_label)
        mamData$x <- mamData[, x_label]
        print(y_label)
        mamData$y <- mamData[, y_label]
        # test if y contains only floats otherwise skip
        if (any(is.na(as.numeric(mamData$y)))) {
            next
        }
        # test if y contains only the same value otherwise skip
        if (length(unique(as.numeric(mamData$y))) == 1) {
            next
        }
        cdat <- comparative.data(data = mamData, phy = mamTree, names.col = "Names")
        m <- pgls(y ~ x, data = cdat, lambda = 1.0)
        s <- summary(m)

        t <- s$coefficients
        # Add column with y_label and x_label
        t <- cbind(t, y_label)
        t <- cbind(t, x_label)
        r2 <- c(s$adj.r.squared, s$r.squared)
        t <- cbind(t, r2)

        regression <- c('intercept', 'slope')
        t <- cbind(t, regression)
        # Add to table the first row
        table_output <- rbind(table_output, t)
    }
}
write.table(table_output, file = args$output_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
