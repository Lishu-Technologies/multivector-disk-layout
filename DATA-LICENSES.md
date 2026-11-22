# Benchmark data and attribution

The software in this repository is licensed under Apache-2.0. The FiQA benchmark uses the following separately published materials.

| Material | Source and version | Label on the data card |
| --- | --- | --- |
| ColBERT token vectors and corpus/query row IDs | [robro612/fiqa_answerai_colbert_small, revision 6fe7f90](https://huggingface.co/datasets/robro612/fiqa_answerai_colbert_small/tree/6fe7f906b70351d889119b771961b85145bb60a1) | CC BY-SA 4.0 |
| FiQA text for the BM25 candidate lists | [BeIR/fiqa, revision 979c07a](https://huggingface.co/datasets/BeIR/fiqa/tree/979c07a7cb5ccc6ca009792241fa1250b98055dd) | See the upstream FiQA data card |

The benchmark scripts download the original inputs from their publishers. The replay archive contains derived candidate lists, document row IDs, MaxSim reference scores and rankings, alongside measurements made by Lishu Technologies. Lists and scores were generated from the cited inputs by pooling embeddings, scoring BM25 candidates and applying exhaustive MaxSim. The FiQA-derived data in the archive is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The archive also contains independently recorded latency and I/O measurements.

The benchmark used the [official Qdrant 1.20.2 release](https://github.com/qdrant/qdrant/releases/tag/v1.20.2). Qdrant is a separate project under its [own Apache-2.0 license](https://github.com/qdrant/qdrant/blob/v1.20.2/LICENSE).
