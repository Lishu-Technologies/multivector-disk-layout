# 多向量磁盘布局工具

面向把 token 向量放在 SSD 上、先召回候选再用 MaxSim 精确重排的团队。输入文档向量、各文档的 token 数和历史候选列表，输出逻辑文档 ID 与磁盘布局位置之间的双向映射。布局规划只需 CPU 和 NumPy。

```sh
git clone https://github.com/Lishu-Technologies/multivector-disk-layout.git
cd multivector-disk-layout
python -m pip install .
multivector-disk-layout documents.npy lengths.npy shortlists.json layout.npz
```

`documents.npy` 是按原文档行号拼接的二维 FP16 token 矩阵。`lengths.npy` 的每项是对应文档的正整数 token 数。`shortlists.json` 是二维列表，每行包含一个历史查询召回的文档行号，可以不等长。输出路径应为新文件。每次发布新的索引快照时，将集合与其映射表一起切换。

## 接入 Qdrant

建立新的磁盘多向量集合。对每个存储位置 `s`，取逻辑文档 `layout.logical_ids(s)` 的向量，以数值 `s` 作为 Qdrant 点 ID 写入。查询时用 `layout.storage_ids(candidate_ids)` 转换候选列表，返回结果时再用 `layout.logical_ids(result_ids)` 还原。你自己的业务 ID 可以继续映射到原文档行号。

在 Qdrant 1.20.2 的本次配置里，只改导入顺序会在整理集合后回到原始 ID 顺序。采用位置 ID 后，57,638 个文档在向量文件中的位置逐篇核验通过。

## 实测适用范围

FiQA 完整语料包含 57,638 篇文档。用均值向量召回 200 个候选时，Qdrant 首次访问的 P95 从 50.46 ms 降至 42.41 ms，降幅 16.0%，95% 配对区间为 8.6%–23.7%。BM25 候选的首访降幅为 2.3%，区间跨零。缓存热起来后，两种布局的延迟基本相同。主实验中两种召回方式各 160 个查询的有序前十均与穷举一致。[复跑入口与原始记录](benchmarks/fiqa/README.md)

从可运行的最小示例开始，可以查看 [Python 快速入门](README.md#install-and-try-it) 或 [Qdrant 最小接入示例](examples/qdrant_demo.py)。源码采用 [Apache-2.0](LICENSE)，FiQA 材料的来源及许可见[数据说明](DATA-LICENSES.md)。
