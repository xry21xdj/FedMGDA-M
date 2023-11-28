#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:34:25 2023

@author: user
"""


"""
This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from utils.utils import *
from utils.constants import *
from utils.args import *

from torch.utils.tensorboard import SummaryWriter
#%%

def init_clients(args_, root_path, logs_root):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_root: path to logs root
    :return: List[Client]
    """
    n_train_samples=1
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators =\
        get_loaders(  # get_loaders in utils
            type_=LOADER_TYPE[args_.experiment],
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation,
            num_workers=args_.num_workers
        )
    '''计算局部迭代次数'''
    for train_iterator in train_iterators:
        n_train_samples=len(train_iterator.dataset)
        print('n_train_samples',n_train_samples)
        
    local_iteration=args_.local_E*int(n_train_samples/args_.bz)
    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue
            
        # for cifar10, at leat two gpu are required
        if args_.experiment=='cifar10':
            if task_id >7:
                args_.device='cuda:1'
        #print('train_iterators',len(train_iterator.dataset))
        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                beta=args_.beta,
                K=local_iteration,
                gamma=args_.gamma
            )

        logs_path = os.path.join(logs_root, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(  #get_client is defined in utils
            client_type=CLIENT_TYPE[args_.method], #defined in constant
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        ) 

        clients_.append(client)
        
    return clients_, local_iteration,n_train_samples


def run_experiment(args_):
    
    test_acc_variance=[]
    test_acc_diff_max=[]
    global_test_acc_all=[]
    torch.manual_seed(args_.seed)
    
    #data file related to the experiment
    data_dir = get_data_dir(args_.experiment) # in utils   

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))#实验结果记录路径
    logs_root += '_R'+ str(args_.n_rounds)+ '_lr'+str(args_.lr) +'_Bz'+str(args_.bz)+ '_LocalE'+str(args_.local_E)+\
        '_SR' +str(args_.sampling_rate) + '_gamma'+str(args_.gamma) +'_beta'+str(args_.beta) + '_mu'+str(args_.mu)
    print("==> Clients initialization..")
    clients, local_iteration, n_train_samples= init_clients(
        args_,
        root_path=os.path.join(data_dir, "train"),
        logs_root=os.path.join(logs_root, "train")
    )
    print('train_samples=',n_train_samples)
    print("==> Test Clients initialization..")
    test_clients,test_local_iteration, n_test_samples= init_clients(
        args_,
        root_path=os.path.join(data_dir, "test"),
        logs_root=os.path.join(logs_root, "test")
    )
    #print('test_samples=',n_test_samples)
    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)
    
    if args_.experiment=='cifar10':
        args_device='cuda:1'
    
    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            beta=args_.beta,
            K=local_iteration,
            gamma=args_.gamma
        )

    # if args_.decentralized:
    #     aggregator_type = 'decentralized'
    # else:
    aggregator_type = AGGREGATOR_TYPE[args_.method] #in constant.py

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            gamma=args.gamma,
            alpha=args_.alpha,
            beta=args_.beta,
            K=local_iteration,
            buffed_clients=args_.buffed_clients,
            poisoned_num=args_.poisoned_num,
            poisoned_gap=args_.poisoned_gap,
            grad_bound=args_.grad_bound,
            epsilon=args_.epsilon,
            delta=args_.delta,
            data_size=n_train_samples,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            local_E=args_.local_E,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            attack_manner=args_.attack_manner,
            scale=args_.scale,
            mean=args_.mean,
            variance=args_.variance,
            add_noise=args_.add_noise,
            poisoned_flag=args_.poisoned_flag,
            seed=args_.seed
        )

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    local_model_seq=[[] for i in range(args_.n_rounds)]
    tot_time=0
    cal_time=[]
    '''construct a two stage update rule'''
    #'''first stage'''
    #aggregator.mix()
    print('the first stage')
    while current_round <=int(args_.n_rounds):
        #torch.cuda.empty_cache()
        variance=0.
        diff_max=0.
        if args_.method=='FedMom':
           print('FedMom')
           aggregator.mix_FedMom()
        elif args_.method=="FedNAG":
           print('FedNAG')
           assert args_.optimizer=='nag_sgd', 'please change the optimizer to nag_sgd'
           aggregator.mix_FedNAG()
        elif args_.method=="FastSlowMo":
           print('FastSlowMo')
           assert args_.optimizer=='fastslowmo_sgd', 'please change the optimizer to fastslowmo_sgd'
           aggregator.mix_FastSlowMo()
        elif args_.method=="DOMO":
              print('DOMO')
              assert args_.optimizer=='domo_sgd', 'please change the optimizer to domo_sgd'
              aggregator.mix_DOMO()
        elif args_.method=="MIME":
              print('MIME')
              assert args_.optimizer=='mime_sgd', 'please change the optimizer to mime_sgd'
              aggregator.mix_MIME()
        elif args_.method=="FedMoS":
              print('FedMos')
              assert args_.optimizer=='fedmos_sgd', 'please change the optimizer to fedmos_sgd'
              aggregator.mix_FedMoS()
        elif args_.method=="FedGLOMO":
              print('FedGLOMO')
              assert args_.optimizer=='fedglomo_sgd', 'please change the optimizer to fedglomo_sgd'
              aggregator.mix_FedGLOMO()
        elif args_.method=="FedAvg-M-Mom":
              print('FedAvg-M-Mom')
              assert args_.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
              aggregator.mix_FedMom()
        elif args_.method=="FedProx":
              print('FedProx')
              assert args_.optimizer=='prox_sgd', 'please change the optimizer to prox_sgd'
              aggregator.mix()
        elif args_.method=="FedCM":
              print('FedCM')
              assert args_.optimizer=='cm_sgd', 'please change the optimizer to cm_sgd'
              aggregator.mix()
        elif args_.method=="FedAdam":
            print('FedAdam')
            aggregator.mix_FedAdam()
        elif args_.method=="FedMGDA-M":
            print('FedMGDA-M')
            assert args_.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
            assert args_.mu ==0, 'please change the global momemtum coefficient to zero'
            aggregator.mix_MGDA_Mom()
        elif args_.method=="FedMGDA-M-Mom":
            print('FedMGDA-M-Mom')
            assert args_.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
            aggregator.mix_MGDA_Mom()
        elif args_.method=="FedMGDA-Mom":
            print('FedMGDA-Mom')
            assert args_.optimizer=='sgd', 'please change the optimizer to sgd'
            aggregator.mix_MGDA_Mom()
        else:
            aggregator.mix()
        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round
        test_acc_variance.append(variance)
        test_acc_diff_max.append(diff_max)
        
    if "save_path" in args_:
        save_root = os.path.join(args_.save_path)

        os.makedirs(save_root, exist_ok=True)
        aggregator.save_state(save_root)
        
    return tot_time,cal_time,local_model_seq,global_test_acc_all, test_acc_variance, test_acc_diff_max

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test_acc_variance=[]
    test_acc_diff_max=[]
    global_test_acc_all=[]
    local_model_seq=[]
    tot_time=0
    cal_time=[]
    args = parse_args()

    #second stage
    run_experiment(args)
