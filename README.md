# Integrating graph genomes and proteomes

Genomics = variant-based phenotype-propensity reference graph to be combined with patient-specific proteomics data --> what can we learn by integrating these?

## Team members

- Friederike Duendar
- Zillur Rahman
- Anita Egebor
- Yan Zhou
- Alvaro Martinez Barrio
- Kumar Koushik Telaprolu 
- Nolan Bruyat

## Architecture Overview

![Architecture overview: genomes and phenotypes build knowledge graphs, which produce embeddings that train a PyG GNN; the GNN and a proteomics knowledge graph feed a GraphRAG LLM that generates insights.](docs/architecture.png)

```mermaid
flowchart LR
    Genomes["Genomes"]
    Phenotypes["Phenotypes"]
    KG["Knowledge Graphs"]
    Embeddings["Embeddings"]
    GNN["GNN (PyG)\n(Encoder)"]
    Proteomics["Proteomics"]
    ProteomicsKG["Knowledge graph"]
    LLM["LLM\n(GraphRAG)"]
    Insights["Insights"]

    Genomes -- "Data + Meta Data" --> KG
    Phenotypes -- "Build schema" --> KG
    KG --> Embeddings
    Embeddings -- "Training model" --> GNN
    GNN --> LLM
    Proteomics --> ProteomicsKG
    ProteomicsKG --> LLM
    LLM --> Insights
```
