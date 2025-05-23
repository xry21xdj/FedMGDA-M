# FedMGDA-M
This repository is the official implementation of : [Accelerating communication-efficient federated multi-task learning with personalization and fairness](https://ieeexplore.ieee.org/document/10552428), published at "IEEE Transactions on Parallel and Distributed Systems" in 2024. The appendix of this paper can be found in research gate [full paper with appendix](https://www.researchgate.net/publication/381310999_Accelerating_Communication-efficient_Federated_Multi-Task_Learning_With_Personalization_and_Fairness).



- we proposes a local momentum based method FedMGDA-M, FedAvg-M , and its derivatives FedAvg-M-Mom, FedMGDA-M-Mom. Besides, we compared them  with other accelerated methods including:
[FedMom](https://arxiv.org/pdf/2002.02090.pdf)
[FedNAG](https://ieeexplore.ieee.org/abstract/document/9891808)
[FedAdam](https://arxiv.org/pdf/2003.00295.pdf)
[DOMO](https://ojs.aaai.org/index.php/AAAI/article/view/20853)
[FastSlowMo](https://ieeexplore.ieee.org/abstract/document/9813376)
[Mime](https://arxiv.org/abs/2008.03606)
[FedMoS](https://liyuqingwhu.github.io/lyq/papers/INFOCOM2023.pdf).
The code for these methods is included in this repository. 

- We also built a tiny real-world test platform with Raspberry Pi 4, as shown in the Figure. If you are interested in implementing the FL in this IoT-based platform, please feel free to reach out: renyou.xie@unsw.edu.au;xierenyou21@163.com;
  
<!--![FL_platform](https://github.com/user-attachments/assets/14f05bb2-f0a5-465a-af64-bf37a9300508) -->

<div align="center">
<img src="https://github.com/user-attachments/assets/14f05bb2-f0a5-465a-af64-bf37a9300508" width="650" height="450" alt="FL Platform">
</div>


## requirement

> Pillow == 8.1.2
> tqdm
> scikit-learn == 0.21.3
> numpy == 1.19.0
> torch == 1.2.0
> matplotlib == 3.1.1
> networkx == 2.5.1
> cvxpy
> torchvision
> tensorboard

## Dataset for experiment

The following table summarizes the datasets and models.

| Dataset  | Task                              | Model                     | distribution                   |
| -------- | --------------------------------- | ------------------------- | ------------------------------ |
| EMNIST   | Handwritten character recognition | 2-layer CNN + 2-layer FFN | pathological split with n=2    |
| CIFAR100 | Image classification              | MobileNet-v2              | pachinko alloaction, alpha=0.1 |
|CIFAR10   | Image Classfication               | VGG16                     | pathological split with n=2    |
|Tiny ImageNet| Image classification           |RestNet18                  |pathological split with n=10    |





See the `README.md` files of respective dataset, i.e., `data/$DATASET`,
for instructions on generating data

## Training

Run on one dataset first to get the data, with a specific  choice of federated learning method.
Specify the name of the dataset (experiment), the used method, and configure all other
hyper-parameters (see hyper-parameters values in the  paper). 



## Evaluation

We give instructions to run experiments on CIFAR-100 dataset as an example
(the same holds for the other datasets). You need first to go to
`./data/cifar100`, follow the instructions in `README.md` to download and partition
the dataset.

### Average performance of personalized models

Run the following scripts, this will generate tensorboard logs that you can interact with to make plots or get the
values.

```eval
# run FedMGDA-M
python run_experiment.py  --experiment cifar100 --method FedMGDA-M --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1  --gamma 1.\
--beta 0.9 --mu 0. --log_freq 2 --device cuda --optimizer m_sgd --seed 1234 --verbose 1

# run FedAvg-M
python run_experiment.py  --experiment cifar100 --method FedMGDA-M --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1  --gamma 1.\
--beta 0.9 --mu 0. --log_freq 2 --device cuda --optimizer m_sgd --seed 1234 --verbose 1

# run FedMom
python run_experiment.py --experiment cifar100 --method FedMom --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01  --local_E 4 --sampling_rate 0.1  --beta 0.9\
 --log_freq 2 --device cuda:0 --optimizer sgd --seed 1234 --verbose 1
 
# run FedNAG
python run_experiment.py --experiment cifar100 --method FedNAG --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01  --local_E 4 --sampling_rate 0.1  --beta 0.9\
 --log_freq 2 --device cuda:0 --optimizer nag_sgd --seed 1234 --verbose 1
 
 # run FedAdam
python run_experiment.py --experiment cifar100 --method FedAdam --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01  --local_E 4 --sampling_rate 0.1  --mu 0.9 \ --beta 0.99 --gamma 0.001
 --log_freq 2 --device cuda:0 --optimizer sgd --seed 1234 --verbose 1
 
# run FedMGDA-Mom
python run_experiment.py --experiment cifar100 --method FedMGDA-M --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01  --local_E 4 --sampling_rate 0.1  --beta 0.9\
 --log_freq 2 --device cuda:0 --optimizer sgd --seed 1234 --verbose 1
 
# run FastSlowMo
python run_experiment.py --experiment cifar100 --method FastSlowMo --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01  --local_E 4 --sampling_rate 0.1 --gamm 0.5 \
--beta 0.5 --log_freq 2 --device cuda:0 --optimizer fastslowmo_sgd --seed 1234 --verbose 1


# run FedMGDA-M-Mom
python run_experiment.py  --experiment cifar100 --method FedMGDA-M-Mom --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1 \
--gamma 1. --beta 0.5 --mu 0.5 --log_freq 2 --device cuda --optimizer m_sgd --seed 1234 --verbose 1

#runFedAvg-M-Mom
python run_experiment.py  --experiment cifar100 --method FedAvg-M-Mom --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1  --gamma 1.\
--beta 0.5 --mu 0.5 --log_freq 2 --device cuda --optimizer m_sgd --seed 1234 --verbose 1

#run DOMO
python run_experiment.py  --experiment cifar100 --method DOMO --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1  --gamma 1.\
--beta 0.5 --mu 0.5 --log_freq 2 --device cuda --optimizer domo_sgd --seed 1234 --verbose 1

#run MIME
python run_experiment.py  --experiment cifar100 --method MIME --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1  --gamma 1.\
--beta 0.1 --mu 0. --log_freq 2 --device cuda --optimizer mime_sgd --seed 1234 --verbose 1

#run FedMoS
python run_experiment.py  --experiment cifar100 --method FedMoS --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1  --gamma 0.9\
--beta 0.5 --mu 0. --log_freq 2 --device cuda --optimizer fedmos_sgd --seed 1234 --verbose 1

# run FedGLOMO
python  run_experiment.py --experiment cifar100 --method FedGLOMO --n_learners 1 --n_rounds 500 --bz 128 --lr 0.01 --local_E 4 --sampling_rate 0.1 --gamma 1.\
--beta 0.9 --mu 0. --log_freq 2 --device cuda --optimizer fedglomo_sgd --seed 1234 --verbose 1

```

## Acknowledgments

I would like to thank the following open-source project for their framework, which I referred to during the development of this project:

- [FedEM](https://github.com/omarfoq/FedEM)

## Citation
If you use our code or wish to refer to our results, please use the following BibTex entry:
```
@ARTICLE{10552428,
  author={Xie, Renyou and Li, Chaojie and Zhou, Xiaojun and Dong, Zhaoyang},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={Accelerating Communication-Efficient Federated Multi-Task Learning With Personalization and Fairness}, 
  year={2024},
  volume={35},
  number={11},
  pages={2239-2253},
  keywords={Data models;Convergence;Internet of Things;Costs;Training;Task analysis;Multitasking;Communication efficiency;data heterogeneity;federated learning;local momentum technique;multi-task learning},
  doi={10.1109/TPDS.2024.3411815}}

```
