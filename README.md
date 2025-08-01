# CacheRAG

This is the repository for the CacheRAG: A Novel Approach to Enhance KG-based RAG Through Caching Mechanisms.

## Install

```console
$ git clone [link to repo]
$ cd CacheRAG-code
$ pip install -r requirements.txt 
```

If you are using Anaconda, you can create a virtual environment and install all the packages:

```console
$ conda create --name CacheRAG python=3.9
$ conda activate CacheRAG
$ pip install -r requirements.txt
```

## Dataset

Set up the mock api server for CRAG, according to the instructions here: https://github.com/ysunbp/CacheRAG-code/tree/main/mockapi

Download data folder, results folder, and utils/CRAG_raw folder from https://drive.google.com/drive/folders/1jNsSfddvmtVhH2edLH8PVdPYhy7qIjTh?usp=sharing

## Experiments

### CRAG
Please follow the these steps to train a summarizer and perform KBQA on CRAG
```console
$ cd scripts
$ python crag-cacherag-deepseek-exp.py
$ python crag-cacherag-deepseek-eval.py
```

### QALD-10-en / CWQ / WebQSP
```console
$ cd sparql-based
$ python auto-gen-sparql.py
$ python -m torch.distributed.run --nproc_per_node 1 CacheRAG-QALD.py
```
