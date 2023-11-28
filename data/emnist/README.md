# EMNIST Dataset

## Introduction

Split EMNIST dataset among `n_clients` as follows:
the dataset will be split as in
  [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629);
  i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes.

## Instructions

### Base usage

For basic usage, `run generate_data.py` with a choice of the following arguments:

- ```--n_tasks```: number of tasks/clients, written as integer
- ```--n_shards```: number of shards
- ``` ---pathological_split```: split manner
- ```--s_frac```: fraction of the dataset to be used; default=``0.2``  
- ```--tr_frac```: train set proportion for each task; default=``0.8``
- ```--val_frac```: fraction of validation set (from train set); default: ``0.0``  
- ```--test_tasks_frac```: fraction of test tasks; default=``0.0``
- ```--seed``` : seed to be used before random sampling of data; default=``12345``


  
## Paper Experiments

In order to generate the data split for Table 2 (Full client participation), run

```
python generate_data.py \
    --n_tasks 100 \
    ---pathological_split
    --n_shards 2 \
    --s_frac 0.2 \
    --tr_frac 0.8 \
    --seed 12345    
```

