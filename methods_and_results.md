# Methods and Results: Federated Proteomics, Haplographs, and Graph Learning

This page presents the chromosome 22 proof-of-concept analysis in a format
that renders directly on GitHub. The figures are ordered from simple
classification summaries to increasingly complex graph views.

## 1. Methods

### Data integration and preprocessing

Three site-specific chromosome 22 proteomics matrices were read with
**pandas** and converted from protein-by-sample to sample-by-protein format.
They were joined to age, sex, site, and binary case/control phenotype
metadata, giving 4,000 samples and 460 measured proteins. **NumPy** was used
for numerical transformations and missing-value handling.

The cohort composition by site was:

| Site | N | Control | Case | Female | Male | Age, mean ± SD (years) |
|---|---:|---:|---:|---:|---:|---:|
| Site 1 | 1,334 | 773 | 561 | 686 | 648 | 52.05 ± 19.60 |
| Site 2 | 1,333 | 791 | 542 | 697 | 636 | 50.88 ± 19.69 |
| Site 3 | 1,333 | 799 | 534 | 674 | 659 | 51.79 ± 19.57 |
| **Total** | **4,000** | **2,363** | **1,637** | **2,057** | **1,943** | **51.57 ± 19.62** |

Here, phenotype 0 is called control and phenotype 1 is called case; sex 0 is
reported as female and sex 1 as male, following the dataset encoding.

Samples were divided into an 80% training set and a 20% held-out test set
using a fixed random seed. Stratification used the joint site-by-phenotype
label, preserving case/control proportions within each site. Protein
filtering was learned from training samples only:

- Detection rate at least 80%
- Median log2 intensity at least 6
- Variance at least 0.01
- For protein pairs with absolute Pearson correlation above 0.8, retain the
  higher-variance protein

### Federated-learning setup

The federated experiment used the same 4,000 samples as the centralized
analysis, distributed across three simulated institutions (`site1`, `site2`,
and `site3`). Site 1 contained 1,334 samples, Site 2 contained 1,333, and
Site 3 contained 1,333. Each site retained its local patient rows and trained
the same binary classifier locally.

A site-by-phenotype stratified 80/20 split produced 3,200 training samples
and 800 held-out test samples. Each site trained only on its local portion of
the training partition; the held-out rows were not used for local updates. The
coordinator initialized the shared coefficient vector and intercept to zero.
The common feature schema used median imputation and standardization learned
from training data.

Training used five communication rounds. In each round, every site performed
one local gradient update using its own training samples and the current global
parameters. With learning rate 0.1 and L2 regularization 1e-4, site `k`
returned updated coefficients and an intercept. The coordinator applied
sample-count-weighted FedAvg:

```text
w_global = sum_k(n_k * w_k) / sum_k(n_k)
b_global = sum_k(n_k * b_k) / sum_k(n_k)
```

The updated global parameters were returned for the next round. The final
global model was evaluated on the untouched 800-sample test partition, with
metrics reported per site and averaged across sites. Only model parameters are
exchanged; raw patient rows and local predictions remain at the sites.

Three federated feature configurations were evaluated:

1. **Proteomics only:** protein abundance measurements.
2. **Proteomics + covariates:** protein abundance, age, and sex.
3. **Proteomics + covariates + haplograph:** the previous features plus
   abundance-weighted graph degree, graph weight, and graph lift summaries.

### Federated PyGCN 2

A federated version of PyGCN 2 used the same complete 30,354-edge undirected
protein graph at every site. Each site trained locally for three epochs per
round; after each of five rounds, the coordinator aggregated the GATv2 model
parameters with sample-count-weighted FedAvg and returned the global parameters
to the sites. Protein abundances, phenotype labels, and covariates remained
site-local; only model parameters were aggregated. The final global model was
evaluated on the same 800 held-out samples.

### Logistic-regression model

The reference classifier was implemented with **scikit-learn**. Median
imputation and standardization were placed inside the model pipeline. A
class-balanced logistic-regression model used protein abundances together with
age, sex, and site. Performance was measured with five-fold stratified
cross-validation on the training set and final evaluation on the untouched
test set.

Reported metrics were accuracy, balanced accuracy, precision, recall, F1
score, ROC AUC, and a confusion matrix.

### Haplograph and knowledge graph

The annotated haplograph edge list was processed with **pandas** and
represented with **NetworkX**. Haploblock co-occurrence edges retained their
observed `weight` and `lift` values. Genomic overlap annotations supplied
haploblock-to-protein membership edges.

Protein nodes were linked to gene symbols and measurement sites and annotated
with:

- Haplograph degree
- Total graph weight
- Total graph lift
- Mean protein abundance
- Detection rate
- Phenotype-associated abundance difference

NetworkX was also used for degree summaries, interpretable subgraph selection,
and network layouts. The full graph is retained in CSV files; figures use labeled high-degree
subsets so that node identities and edge statistics remain readable.

### PyTorch Geometric comparison

Two graph-neural-network models were implemented with **PyTorch** and
**PyTorch Geometric (PyG)**. PyGCN 1 used two `GCNConv` layers and a 2,000-edge
protein projection. PyGCN 2 used the complete 30,354-edge undirected protein
projection (60,708 directed edges), two `GATv2Conv` attention layers, and both
projected edge attributes: aggregated haplograph weight and lift. To avoid
replicating the full graph 4,000 times in GPU memory, PyGCN 2 encoded the
shared protein graph once per optimization step and pooled the learned protein
embeddings with each patient's abundance vector; age, sex, and site were then
included in the classifier. PyGCN 2 used three candidate hyperparameter sets,
selected the best by a stratified validation split, trained with early
stopping, and used the same held-out test split and five-fold training-set
cross-validation as the logistic-regression reference.
PyGCN 1 was therefore a smaller two-layer GCN with a 2,000-edge graph,
without attention or edge attributes, whereas PyGCN 2 used two attention-based
GATv2Conv layers with hidden dimensions and attention heads selected by tuning
on the validation split. PyGCN 2 also used the complete projected graph,
weight/lift edge features, AdamW optimization, dropout, gradient clipping, and
early stopping; its shared-graph pooling strategy reduced memory use compared
with creating a separate full graph for every patient.

## 2. Results

### Logistic-regression performance

| Metric | Five-fold CV mean | Five-fold CV SD | Held-out test |
|---|---:|---:|---:|
| Accuracy | 0.851 | 0.018 | 0.855 |
| Balanced accuracy | 0.847 | 0.019 | 0.848 |
| Precision | 0.815 | 0.027 | 0.833 |
| Recall | 0.823 | 0.027 | 0.807 |
| F1 score | 0.819 | 0.022 | 0.820 |
| ROC AUC | 0.930 | 0.011 | 0.933 |

The held-out confusion matrix contained 420 true negatives, 53 false
positives, 63 false negatives, and 264 true positives.

### Federated classification comparison

The following values are means across the three sites after five FedAvg rounds;
800 held-out samples were evaluated in total.

| Feature configuration | Accuracy | Balanced accuracy | F1 score | ROC AUC | ROC AUC difference vs proteomics-only |
|---|---:|---:|---:|---:|---:|
| Proteomics only | 0.892 | 0.897 | 0.875 | 0.962 | 0.0000 |
| Proteomics + age + sex | 0.890 | 0.895 | 0.872 | 0.962 | +0.0003 |
| Proteomics + age + sex + haplograph | 0.892 | 0.897 | 0.875 | 0.962 | +0.0001 |

Using all 4,000 samples with a held-out evaluation removes the earlier
training-set optimism. Proteomics alone gave the strongest or essentially tied
performance. Adding age, sex, and haplograph summaries changed mean ROC AUC by
less than 0.001, indicating that the synthetic proteomic signal dominates this
federated task.

### Centralized versus federated models

| Model | Training mode | Test accuracy | Test balanced accuracy | Test F1 | Test ROC AUC |
|---|---|---:|---:|---:|---:|
| Logistic regression | Centralized | 0.855 | 0.848 | 0.820 | 0.933 |
| PyGCN 2 | Centralized | 0.469 | 0.513 | 0.537 | 0.517 |
| Logistic regression | Federated FedAvg | 0.892 | 0.897 | 0.875 | 0.962 |
| PyGCN 2 | Federated FedAvg | 0.560 | 0.496 | 0.214 | 0.557 |

The federated PyGCN 2 model had higher ROC AUC than centralized PyGCN 2, but
remained substantially weaker than logistic regression. Optimization differs
between the approaches, so these values describe this implementation rather
than a universal advantage for centralized or federated training.

### Graph size and graph/model comparison

| Graph or comparison quantity | Result |
|---|---:|
| Haploblock nodes | 4,344 |
| Haploblock co-occurrence edges | 187,030 |
| Protein nodes | 397 |
| Gene nodes | 391 |
| Site nodes | 3 |
| Integrated graph edges | 491,898 |
| Proteins compared with logistic regression | 397 |
| Top-20 graph/model overlap | 2 proteins (10%) |
| Spearman degree vs absolute coefficient | −0.088 |
| Spearman lift vs absolute coefficient | −0.087 |

The low overlap and near-zero correlations indicate that graph centrality and
predictive proteomic importance are complementary. A highly connected protein
is central to genomic co-occurrence structure, but it is not necessarily the
protein that best distinguishes the phenotype.

### PyG versus logistic regression

| Model | Test ROC AUC | CV ROC AUC | Test balanced accuracy | Test F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.933 | 0.930 ± 0.011 | 0.848 | 0.820 |
| PyGCN 1 | 0.512 | 0.496 ± 0.018 | 0.500 | 0.580 |
| PyGCN 2 | 0.517 | 0.535 ± 0.026 | 0.513 | 0.537 |

PyGCN 2 improved over PyGCN 1 in cross-validation ROC AUC (0.535 versus
0.496) and balanced accuracy (0.506 versus 0.497), but both graph models were
near chance and substantially below logistic regression. The graph models may
need a different patient-level graph formulation, richer node features, and
more task-specific architecture design before they can exploit the network
structure effectively for phenotype prediction.

## 3. Figures

### Figure 1 — Held-out test confusion matrix

![Figure 1: Held-out test confusion matrix](proteomics/logistic_regression_results/test_confusion_matrix.png)

The horizontal axis shows the model prediction, while the vertical axis shows
the true phenotype. Each cell is a number of samples: diagonal cells are
correct predictions and off-diagonal cells are errors. The figure is
descriptive and does not by itself provide a p-value or prove statistical
significance.

### Figure 2 — Largest logistic-regression feature coefficients

![Figure 2: Largest logistic-regression feature coefficients](proteomics/logistic_regression_results/top_model_features.png)

The horizontal axis is the standardized logistic coefficient and the vertical
axis lists proteins or covariates. Bars to the right increase predicted case
probability and bars to the left decrease it; longer bars indicate stronger
model association after adjustment. These are model coefficients, not
p-values, so they should not be called statistically significant without
confidence intervals or a formal multiple-testing analysis.

### Figure 3 — Most connected proteins in the haplograph

![Figure 3: Most connected proteins](proteomics/knowledge_graph/top_proteins_by_graph_degree.png)

The horizontal axis is haplograph degree and the vertical axis lists proteins
or gene symbols. A longer bar means that the protein is associated with more
annotated haplograph edge records. This indicates network connectivity, not
higher expression or stronger phenotype prediction; no significance test is
shown.

### Figure 4 — Haplograph connectivity versus phenotype signal

![Figure 4: Connectivity versus phenotype signal](proteomics/knowledge_graph/graph_degree_vs_phenotype_signal.png)

Each point is one protein. The x-axis shows haplograph degree on a logarithmic
scale, and the y-axis shows the mean log2 expression difference between
phenotype 1 and phenotype 0. Point color represents the absolute injected
phenotype effect; stronger color means a larger simulated effect. This is an
exploratory visualization, not a statistical significance test.

### Figure 5 — Protein-labeled haplograph network

![Figure 5: Haplograph-only network](proteomics/graph_comparison/haplograph_network.png)

Each node is labeled only with its mapped protein name(s). Larger nodes have
higher degree in the displayed subgraph; no cluster or haploblock identifiers
are displayed. Each edge is labeled with `w` (co-occurrence
weight) and `l` (lift); edge width also represents the logarithm of weight,
while edge color represents lift. Higher lift means stronger co-occurrence
relative to the independence expectation. Lift greater than 1 is an enrichment
measure, but this figure does not show p-values or confidence intervals.

### Figure 6 — Integrated haploblock–proteomics network

![Figure 6: Integrated haploblock–proteomics network](proteomics/graph_comparison/haplograph_proteomics_network.png)

Squares represent haploblocks labeled with their mapped protein name(s), and
circles represent labeled proteins. Node size is proportional to haplograph
degree: larger squares are more connected haploblocks and larger circles are
more connected proteins. Grey membership lines connect proteins to haploblocks;
their thickness is proportional to the mapped protein's log-scaled haplograph
degree. Orange lines show haploblock co-occurrence, and their thickness
combines the log-scaled edge weight and lift. The legend explains node size,
line types, and line thickness. Protein color represents the difference in
mean log2 abundance between phenotype groups. Colors, node sizes, and line
widths are visual encodings, not statistical significance tests.

### Figure 7 — Graph connectivity versus logistic-regression importance

![Figure 7: Graph connectivity versus logistic-regression importance](proteomics/graph_comparison/graph_vs_logistic_regression.png)

Each point is one matched protein. The x-axis is haplograph degree on a
logarithmic scale and the y-axis is the absolute standardized logistic
coefficient, also on a logarithmic scale. Points higher up have stronger
model influence regardless of direction, while points farther right are more
network-connected. The weak correlation and low top-20 overlap show different
rankings, but the plot alone does not establish statistical significance.

## 4. Reproducibility and files

Main scripts:

- [`scripts/analyze_proteomics.py`](scripts/analyze_proteomics.py)
- [`scripts/build_knowledge_graph.py`](scripts/build_knowledge_graph.py)
- [`scripts/plot_haplograph_comparison.py`](scripts/plot_haplograph_comparison.py)
- [`scripts/run_pyg_comparison.py`](scripts/run_pyg_comparison.py)
- [`scripts/run_federated_pygcn.py`](scripts/run_federated_pygcn.py)

Result directories:

- [`proteomics/logistic_regression_results/`](proteomics/logistic_regression_results/)
- [`proteomics/knowledge_graph/`](proteomics/knowledge_graph/)
- [`proteomics/graph_comparison/`](proteomics/graph_comparison/)
- [`proteomics/pyg_results/`](proteomics/pyg_results/)
- [`proteomics/federated_pyg_results/`](proteomics/federated_pyg_results/)
