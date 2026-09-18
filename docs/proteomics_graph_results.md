# Proteomics–haplograph results

The filtered logistic-regression model showed good discrimination of the
synthetic case/control phenotype. Five-fold cross-validation on the training
set gave ROC AUC 0.930 ± 0.011, balanced accuracy 0.847 ± 0.019, and F1 score
0.819 ± 0.022. On the untouched stratified test set, ROC AUC was 0.933,
balanced accuracy was 0.848, and F1 score was 0.820. The confusion matrix
contained 420 true negatives, 53 false positives, 63 false negatives, and 264
true positives.

The full annotated haplograph contained 4,344 haploblock nodes and 187,030
co-occurrence edges in the plotted input, with edge width representing
co-occurrence weight and edge color representing lift. The integrated
haplograph–proteomics network added 397 protein nodes, 303,284
haploblock–protein membership edges, 391 gene nodes, and site measurement
links. To compare network structure with prediction, we matched 397 proteins
to their absolute protein-level logistic coefficients. The 20 most connected
proteins and the 20 proteins with the largest absolute model coefficients
overlapped by 2 proteins (10%); Spearman correlations were -0.088 for degree
versus absolute coefficient and -0.087 for lift versus absolute coefficient.
Thus, in this simulated dataset, haplograph connectivity and predictive
importance capture complementary properties rather than the same ranking:
network centrality describes genomic co-occurrence structure, whereas the
classifier prioritizes proteomic signal conditional on age, sex, and site.

As a graph-neural-network comparison, a PyTorch Geometric two-layer GCN was
trained on patient-specific graphs containing the same 397 proteins and a
2,000-edge high-weight protein projection of the haplograph. The PyG model
achieved held-out ROC AUC 0.512, balanced accuracy 0.500, and F1 score 0.580;
five-fold cross-validation gave ROC AUC 0.496 ± 0.018 and balanced accuracy
0.494 ± 0.015. In contrast, the logistic-regression model achieved held-out
ROC AUC 0.933 and cross-validation ROC AUC 0.930 ± 0.011. The current GCN
therefore underperformed the simpler model, likely because this small,
lightly trained baseline compresses each patient's node-level abundance
pattern using global pooling and does not yet use attention, edge attributes,
hyperparameter tuning, or a larger projected graph. It is useful as a
reproducible PyG baseline, but the logistic model is the stronger result for
the present dataset.
