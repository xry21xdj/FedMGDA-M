 # CIFAR10 Dataset

## Introduction

split as emnist did
  
## Paper Experiments

python generate_data.py \
    --n_tasks 10 \
    ---pathological_split\
    --n_shards 2 \
    --s_frac 0.2 \
    --tr_frac 0.8 \
    --seed 12345    
