# ProGenome
*A Federated Workflow for Genome Graph and Proteomic Integration*

📊 **Presentation slides:** [Team 3: ProGenome (Google Slides)](https://docs.google.com/presentation/d/13wHiF-xPHeDyy7XYMmTSwxAYUswgSutH7x7dZ02SOOs/edit) · 📄 **Genomics methods and results:** [genomics/RESULTS.md](genomics/RESULTS.md)

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.8-3C2179)](https://pyg.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](genomics/Dockerfile)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA%20FLARE-2.9%20FedAvg-76B900?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/NVFlare)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Nemotron%203%20Super-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![NVIDIA Brev](https://img.shields.io/badge/NVIDIA%20Brev-A100%2080GB-76B900?logo=nvidia&logoColor=white)](https://brev.nvidia.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.26%20community-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.6%20%2B%20nx--cugraph-1B6AC6)](https://networkx.org/)
[![Data](https://img.shields.io/badge/Data-1000G%20HaploGraph%20chr22-0E7C7B)](https://data.haploblocks.org/haplograph/1000G/)
[![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen?logo=pytest&logoColor=white)](genomics/tests/)
[![Hackathon](https://img.shields.io/badge/Nordic%20Biobank%20x%20NVIDIA-Federated%20Learning%20Hackathon%202026-5A9E3F)](https://github.com/collaborativebioinformatics/ProGenome)


## 🎯 Our Mission
To develop a federated workflow that integrates haploblock-based genome-graph with proteomic data from each participating institution.


<img width="711" height="384" alt="image" src="https://github.com/user-attachments/assets/fc0c51d3-af4f-4afe-bfbb-a674b07fac2c" />

## 🚀 Quick Start (Demo)

The initial proof-of-concept demonstration will focus on chromosome 22, a test case before extending the workflow to additional chromosomes or the whole genome.

The demo will integrate haplotype information, gene information, and proteomics.

### Required Datasets

We build upon the work of previous hackathons, documented at <haploblocks.org>
From there, we leverage a graph that encodes how haploblock clusters co-occur across individuals

![Haploblock co-occurrence graph](https://haploblocks.org/figures/haploblock_co_occurence_graph.png)


1. **Haploblock BED file**

   Defines the genomic coordinates and identifiers of the predefined haploblocks on chromosome 22.

   Within each haploblock, an individual's haploblock hashes will be represented (we build upon the ideas and data output from the [HaploBlock HPC pipeline project](https://github.com/MauricioMoldes/haploblocks-hpc)). These hashes provide compact identifiers that allow haplotype patterns to be compared across participants.
   Each node represents a **haploblock cluster**, i.e. haploblocks with similar genetic variants across multiple individuals.

3. **Gene BED file**

   Defines the genomic coordinates of genes located on chromosome 22. Genomic-coordinate overlap will be used to determine which genes fall within or overlap each haploblock.

4. **Gene-to-protein mapping**

   Connects chromosome 22 genes to their corresponding protein identifiers.

5. **Proteomic data**

   Proteomic data for proteins encoded by genes located on chromosome 22.*

   Before connecting to the real genomic haploblock graph, we built a synthetic proteomics dataset to validate our data structure and analysis pipeline end-to-end.

### *What we generated
   - Protein intensity values (log2 scale) for all ~460 unique proteins encoded by genes on chromosome 22
   - 3 simulated hospital sites, 4000 patients total
   - For every patient: age, sex, and a binary phenotype (case/control)
   - A deliberately injected signal: each protein's intensity depends, to varying degrees, on the patient's age, sex, and phenotype — some proteins strongly, most only weakly, to mimic real biological heterogeneity
   
   **Result:** the injected phenotype effect is clearly recoverable, and consistent across all 3 simulated sites:
   
   ![Protein intensity by phenotype and site](proteomics/plots/top_phenotype_protein_boxplot.png)
   
   A broader view across the 40 proteins most influenced by age/sex/phenotype shows visible structure separating the two phenotype groups:
   
   ![Heatmap of top signal proteins](proteomics/plots/proteomics_heatmap_top_signal.png)
   
   This confirms the proteomics side of the pipeline behaves as expected, and gives us matched patient-level data (age, sex, phenotype, protein intensities) ready to be connected to the genomic haploblock hashes.

### Data Integration

The integrated graph will represent relationships among participants, haplotype patterns, haploblocks, genes, and proteins:

```mermaid
flowchart LR
    P["Participant"] --> H["Haploblock hash"]
    H --> B["Haploblock"]
    B --> R["Encoded protein"]
    P --> A["Measured protein abundance"]
    R --> A
```

### Reproducible proteomics analysis

The new 4,000-sample synthetic matrices can be analyzed with:

```bash
python scripts/analyze_proteomics.py
python scripts/build_knowledge_graph.py
```

The analysis writes filtered-protein lists, five-fold cross-validation and
held-out test metrics, confusion-matrix and coefficient figures to
`proteomics/logistic_regression_results/`. The graph script writes node/edge
tables and a top-protein visualization to `proteomics/knowledge_graph/`.
Pass `--edges path/to/edges_lift_above_threshold_uniprot.csv.gz` to the graph
script when the annotated haplograph edge file is available.

For example, the graph may represent that a participant carries a particular haplotype pattern within a chromosome 22 haploblock, that the gene encodes a particular protein, and that the participant has a measured abundance value for that protein.

---
## Team members

- Friederike Duendar
- Zillur Rahman
- Anita Egebor
- Yan Zhou
- Alvaro Martinez Barrio
- Kumar Koushik Telaprolu 
- Nolan Bruyat
