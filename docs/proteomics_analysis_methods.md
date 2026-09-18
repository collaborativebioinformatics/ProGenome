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

All preprocessing and tabular data integration were performed with Python
using pandas and NumPy. Logistic regression, train/test splitting, filtering
support, cross-validation, and performance metrics were implemented with
scikit-learn. Matplotlib and seaborn were used for the confusion matrix,
coefficient, and summary plots. NetworkX was used to represent the
haploblock–protein knowledge graph, calculate network summaries, select
interpretable subgraphs, and lay out the network visualizations. PyTorch
Geometric (PyG) was not used for the original logistic-regression or
knowledge-graph construction workflow; it was added separately for the
graph-neural-network comparison described below.

For the graph-neural-network comparison, PyTorch and PyTorch Geometric were
used. PyGCN 1 used two GCNConv layers and a 2,000-edge projected protein graph.
PyGCN 2 used the complete projected protein graph (30,354 undirected edges),
GATv2Conv attention layers, and aggregated haplograph weight and lift as edge
attributes. PyGCN 2 encoded the shared graph once per optimization step and
used each patient's abundance vector to pool protein embeddings, avoiding
memory-expensive duplication of the full graph. Three hyperparameter settings
were compared on a validation split, followed by early-stopped training and
five-fold cross-validation using the same site-by-phenotype stratification as
the logistic-regression model.
