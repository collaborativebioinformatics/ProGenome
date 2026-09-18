# Proteomics analysis and knowledge graph methods

Protein intensities from the three site-specific chromosome 22 matrices were
transposed to sample-by-protein format and joined to age, sex, site, and
case/control phenotype metadata. We used an 80% training set and a 20%
held-out test set, stratified on the joint site-by-phenotype label to preserve
case/control balance within each site. All feature filtering was learned from
training samples only: proteins had to be observed in at least 80% of training
samples, have median log2 intensity at least 6, and have variance at least
0.01; for pairs with absolute Pearson correlation above 0.8, the
higher-variance protein was retained. A class-balanced logistic-regression
model used median imputation, standardization, and age, sex, and site
covariates. Generalization was estimated by five-fold stratified
cross-validation on the training set and by final evaluation on the untouched
test set using accuracy, balanced accuracy, precision, recall, F1 score, ROC
AUC, and a confusion matrix.

The knowledge graph was constructed from the annotated haplograph edge list.
Haploblock co-occurrence edges retain their observed weight and lift values;
haploblock-to-protein membership edges are derived from genomic overlap, and
protein nodes are linked to gene symbols and the three measurement sites.
Protein nodes additionally store mean intensity, detection rate,
phenotype-associated intensity difference, and haplograph degree, weight, and
lift. For visualization, the full graph is summarized using the highest-degree
nodes rather than randomly subsampling edges.
