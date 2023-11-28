
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 20:11:09 2022

@author: user
"""

import os
import time
#import rm

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import numpy.linalg as LA
import heapq

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from utils.torch_utils import *
from utils.min_norm_solvers import *
from torch.autograd import Variable
import copy
class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: List[Client]

    test_clients: List[Client]

    global_learners_ensemble: List[Learner]

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients:

    n_learners:

    clients_weights:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_train_logger:

    global_test_logger:

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__
    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr,
            mu,
            gamma,
            alpha,
            beta,
            K, #local update step
            buffed_clients,
            poisoned_num,
            poisoned_gap,
            grad_bound,
            epsilon,
            delta,
            data_size,
            attack_manner='A1',
            scale=1, #byzatine attack variable
            mean=0, #byzantine attack variable
            variance=0.001, #byzantine attack variable
            sampling_rate=1.,
            local_E=1,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            add_noise=False,
            poisoned_flag=False,
            *args,
            **kwargs
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        #self.device = self.global_learners_ensemble.device
        self.cuda_for_split= self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.model_dim = self.global_learners_ensemble.model_dim
        print('model_dim=',self.model_dim)

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)
        
        
        self.local_E=local_E
        #**************************************
        # poisoned variables
        
        self.poisoned_num=poisoned_num
        self.poisoned_flag=poisoned_flag
        self.poisoned_gap=poisoned_gap
        self.attack_manner=attack_manner
        self.scale=scale
        self.mean=mean
        self.variance=variance
        
        #*********************************************
        np.random.seed(123)
        self.alpha=alpha

         # global learning rate
        #******************************************************
        
        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0
        
        self.diff=[]
        self.last_param=[]
        self.write_logs()
        
        #parameter for DP
        
        self.K=K
        

        
       # parameter for FedMom
        self.v_old=copy.deepcopy(self.global_learners_ensemble[0].model.state_dict(keep_vars=True))
        # for key in self.v_old:
        #     self.v_old[key].data.fill_(0.)
        
        #parameter for FedNAG, FastSlowMo, DOMO
        self.buffer=None
        self.old_model=None
        self.global_buffer =copy.deepcopy(self.global_learners_ensemble[0].model.state_dict(keep_vars=True))
        
        # parameter for FedGLOMO
        self.old_model = copy.deepcopy(self.global_learners_ensemble[0].model.state_dict(keep_vars=True))
        self.flag = 0
        
        #parameter for FedAdam
        self.v_old_adam=copy.deepcopy(self.global_learners_ensemble[0].model.state_dict(keep_vars=True))
        
        self.m_old=copy.deepcopy(self.v_old_adam)
        for key in self.v_old:
            self.v_old_adam[key].data.fill_(0.)
            self.m_old[key].data.fill_(0.)
            
        self.mu=mu
        self.gamma_g=gamma 
        self.beta=beta
        
        print('alpha=',alpha,'mu=',mu,'  gamma=',gamma, 'beta=',beta)
        
        #parameter for FedAGNL
        self.w_old=copy.deepcopy(self.global_learners_ensemble[0].model.state_dict(keep_vars=True))
        
    def get_max_index(self,num_list,topk=10):
        tmp_list=copy.deepcopy(num_list)
        tmp_list.sort()
        max_num_index=max_num_index=[num_list.index(one) for one in tmp_list[::-1][:topk]]
        return max_num_index
        #min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))

    def get_num_positive(self,num_list):
        count_num=0
        for i in range(len(num_list)):
            if num_list[i]>=0:
                count_num+=1
        return count_num

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()

    def write_logs(self):
        self.update_test_clients()

        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.

           
            
            global_test_loss = 0.
            global_test_acc = 0.


            
            total_n_samples = 0
            total_n_test_samples = 0
            train_samples_list=[]
            test_samples_list=[]

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    # with np.printoptions(precision=3, suppress=True):
                    #     print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples
                
                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples
                train_samples_list.append(client.n_train_samples)
                test_samples_list.append(client.n_test_samples)
                
            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples #average
            

            
                   
            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)

                        
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)


        if self.verbose > 0:
            print("#" * 80)
  
    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            chkpts_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            weights = np.load(chkpts_path)

            for client_id, client in enumerate(clients):
                client.learners_ensemble.learners_weights = weights[client_id]

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=self.n_clients_per_round)
            print('sampled_clients=',len(self.sampled_clients))


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
   
    ''' byzantine attack moduel'''
    def byzantine_attack(self,attack_manner,global_learners_ensemble,sampled_clients,
                         state_dict_initial, scale, mean, variance,client_id):
        '''attack manner 1: w_t^k-w_t=-w_t'''
        print('attack manner=',attack_manner)
        if attack_manner=='A1':
            for learner_id, learner in enumerate(global_learners_ensemble):
                learner = sampled_clients[0].learners_ensemble[learner_id]     
                poisoned_state_dict = learner.model.state_dict(keep_vars=True)
                
                for key in poisoned_state_dict:
                    # poisoned strategy 
                    sampled_clients[client_id].learners_ensemble[learner_id].model.state_dict(keep_vars=True)[key].data=(- state_dict_initial[key].data.clone())
    
        '''attack manner 2: w_t^k-w_t=C(w_t^k-w_t)'''
        if attack_manner=='A2':
            
            for learner_id, learner in enumerate(global_learners_ensemble):
                learner = sampled_clients[0].learners_ensemble[learner_id]     
                poisoned_state_dict = learner.model.state_dict(keep_vars=True)
                
                for key in poisoned_state_dict:
                    # poisoned strategy 
                    sampled_clients[client_id].learners_ensemble[learner_id].model.state_dict(keep_vars=True)[key].data=(scale*
                      sampled_clients[client_id].learners_ensemble[learner_id].model.state_dict(keep_vars=True)[key].data.clone())
 
        '''attack manner 3: Gaussian noise'''
        if attack_manner=='A3':
            for learner_id, learner in enumerate(global_learners_ensemble):
                learner = sampled_clients[0].learners_ensemble[learner_id]     
                poisoned_state_dict = learner.model.state_dict(keep_vars=True)
                
                for key in poisoned_state_dict:
                    # poisoned strategy 
                    sampled_clients[client_id].learners_ensemble[learner_id].model.state_dict(keep_vars=True)[key].data=(
                        gaussian_noise_attack(poisoned_state_dict[key].data.shape, mean, variance, device=learner.device))
                      
   
    
   
    
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def mix(self):
        '''
        averaging aggregation
        '''
        
        self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        print('local_E=',self.local_E)
        'sampling clients'
        self.sample_clients()
        #print(self.sampled_clients)
        
 
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        #self.weights= 1/len(self.sampled_clients)*torch.ones(len(self.sampled_clients))
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()

        print('weights=',self.weights)
        for learner_id, learner in enumerate(self.global_learners_ensemble):
        
            self.diff = average_learners_FedAvg(self.sampled_clients, learner, self.weights,self.gamma_g, learner_id)
               
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def mix_FedMom(self):
        '''parameter: eta=[1,M/S], S is the number of sampled clients
        :beta, momentum coefficient'''
        
        ''' code for <<faster on-device training using new federated momentum algorithm>>'''
        print('local_E=',self.local_E)
        'sampling clients'
        

        self.sample_clients()
        #print(self.sampled_clients)
        
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        self.last_param=self.global_learners_ensemble[0].model.parameters()
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        #self.weights= 1/len(self.sampled_clients)*torch.ones(len(self.sampled_clients))
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()
        #eta=1/self.sampling_rate #[1,M/S]
        eta=1
        #print('weights=',self.weights) 
              
        print('\n eta=',eta)

        learner =    self.global_learners_ensemble[0]     
        self.diff, self.v_old=average_learners_FedMom(self.sampled_clients, learner, self.weights,self.v_old, eta, self.beta)
        

        # assign the updated model to all clients
        self.update_clients()
        
        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
    
    def mix_FedNAG(self):
        '''gamma,'''
        
        ''' code for <<Federated learning with nesterove accelerated gradient>>'''
        print('local_E=',self.local_E)
        'sampling clients'
        

        self.sample_clients()
        #print(self.sampled_clients)
        
        # allocate the global model to cuda 1
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        self.last_param=self.global_learners_ensemble[0].model.parameters()
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        #self.weights= 1/len(self.sampled_clients)*torch.ones(len(self.sampled_clients))
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()

        
        # copy the smapled model
        learner =    self.global_learners_ensemble[0]   
        self.buffer=average_learners_FedNAG(self.sampled_clients, learner, self.weights)
            

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()  

            
    def mix_FedAdam(self):
        '''gamma,'''
        
        ''' code for <<adptive federated optimization>>'''
        print('local_E=',self.local_E)
        'sampling clients'
        

        self.sample_clients()
        #print(self.sampled_clients)
        
        # allocate the global model to cuda 1
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        self.last_param=self.global_learners_ensemble[0].model.parameters()
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        #self.weights= 1/len(self.sampled_clients)*torch.ones(len(self.sampled_clients))
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()


        # copy the smapled model

        learner = self.global_learners_ensemble[0]
        self.m_old,self.v_old_adam=average_learners_FedAdam(self.sampled_clients, learner, self.weights, \
                            self.m_old,self.v_old_adam, self.mu, self.beta, self.gamma_g)
               

        
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()  



    def mix_FastSlowMo(self):
        '''gamma,'''
        
        ''' code for <<FastSlowMo>>'''
        print('local_E=',self.local_E)
        'sampling clients'
        

        self.sample_clients()
        #print(self.sampled_clients)
        
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        
        self.last_param=self.global_learners_ensemble[0].model.parameters()
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        #self.weights= 1/len(self.sampled_clients)*torch.ones(len(self.sampled_clients))
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()

        
        self.buffer,self.global_buffer=average_learners_FastSlowMo(self.sampled_clients, self.global_learners_ensemble[0], self.weights, self.global_buffer, self.beta)
               
        
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()  
            
            
            
    def mix_DOMO(self):
        
        ''' code for <<DOMO>>'''
        print('local_E=',self.local_E)
        'sampling clients'
        
        # load the global model to the selected cuda
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        
        self.sample_clients()
        
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()

        learner = self.global_learners_ensemble[0]
        for p in learner.optimizer.param_groups:
               self.lr=p['lr']
               print('lr=',self.lr)
        
        self.buffer, self.old_model=average_learners_DOMO(self.sampled_clients, learner, self.weights, self.buffer, self.old_model, 
                                              self.K, self.lr, self.gamma_g, self.beta, self.mu)
               

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
        
        # premomentum update
        self.update_buffer_DOMO()
            
    
    #%%%%%%%
    def mix_MIME(self):
        '''gamma,'''
       
        ''' code for <<MIME>>'''
    
        # load the global model to the selected cuda
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        print('local_E=',self.local_E)
        'sampling clients'
       

        self.sample_clients()
        #print(self.sampled_clients)
        for client in self.sampled_clients:
            client.compute_full_gradients()
            
            # load the full gradient to optimizer
            client.learners_ensemble[0].optimizer.set_full_gradient_MIME()
    
    
        self.last_param=self.global_learners_ensemble[0].model.parameters()
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
       
        #self.weights= 1/len(self.sampled_clients)*torch.ones(len(self.sampled_clients))
        self.weights =\
           torch.tensor(
               [client.n_train_samples for client in self.sampled_clients],
               dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()
       
        learner = self.global_learners_ensemble[0]
        self.buffer=average_learners_MIME(self.sampled_clients, learner, self.weights, self.buffer,self.beta)
              
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()      

    def mix_FedMoS(self):
        
        ''' code for <<FedMoS>>'''
        
        # load the global model to the selected cuda
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        print('local_E=',self.local_E)
        'sampling clients'
        self.sample_clients()

        for client in self.sampled_clients:
            client.compute_full_gradients()
            
            # load the full gradient to optimizer
            client.learners_ensemble[0].optimizer.set_initial_buffer_FedMoS()        
        
        
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        #get the weights
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()

            
        learner = self.global_learners_ensemble[0]
        for p in learner.optimizer.param_groups:
            self.lr=p['lr']
            print('lr=',self.lr)
        
        self.buffer=average_learners_FedMoS(self.sampled_clients, learner, self.weights, self.buffer, self.K, self.lr,  self.beta)
               
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()      
                                   
    def mix_FedGLOMO(self):
        
        #torch.cuda.empty_cache()
        import subprocess

        # 运行nvidia-smi命令并获取输出
        output = subprocess.check_output(['nvidia-smi'])

        # 打印GPU占用率信息
        print('before aggregation', output.decode('utf-8'))
        ''' code for <<FedGLOMO>>'''
        print('local_E=',self.local_E)
        'sampling clients'
        self.sample_clients()
        self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(self.cuda_for_split)
        # update the old model
        old_models_all=[]
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            for client in self.sampled_clients:
                #get the old global model
                client.learners_ensemble[learner_id].model.load_state_dict(self.old_model)
                for i in range(self.local_E):
                    client.step()    # perform one step for the client
                #collect the old model parameters
                old_models_all.append(copy.deepcopy(client.learners_ensemble[learner_id].model).to(self.cuda_for_split).state_dict(keep_vars=True))
    
        
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            for client in self.sampled_clients:
                #get the old global model
                client.learners_ensemble[learner_id].model.load_state_dict(self.global_learners_ensemble[0].model.state_dict())
                for i in range(self.local_E):
                    client.step()    # perform one step for the client
                #collect the old model parameters
                # new_models_all.append(client.model.state_dict(keep_vars=True))
    
    
        #get the weights
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()

        learner = self.global_learners_ensemble[0]

        self.old_model, self.buffer=average_learners_FedGLOMO(self.sampled_clients, learner, self.weights, self.beta, old_models_all, self.old_model, self.buffer)
         # 运行nvidia-smi命令并获取输出
        output = subprocess.check_output(['nvidia-smi'])

        # 打印GPU占用率信息
        print('after aggregation', output.decode('utf-8'))      

        
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()       
            
    ########################################
    def mix_FedAGNL(self):
        '''gamma,'''
        
        ''' code for <<adptive federated optimization>>'''
        print('local_E=',self.local_E)
        'sampling clients'
        

        self.sample_clients()
        #print(self.sampled_clients)
        
        self.last_param=self.global_learners_ensemble[0].model.parameters()
        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step()    # perform one step for the client
        
        #self.weights= 1/len(self.sampled_clients)*torch.ones(len(self.sampled_clients))
        self.weights =\
            torch.tensor(
                [client.n_train_samples for client in self.sampled_clients],
                dtype=torch.float32)

        self.weights = self.weights / self.weights.sum()


        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            for p in learner.optimizer.param_groups:
                self.lr=p['lr']
                print('lr=',self.lr)
            self.w_old, self.diff =average_learners_FedAGNL(learners,learner,\
                self.weights,self.w_old,self.alpha,self.beta,self.gamma_g,self.mu)


        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()  

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                self.global_learners_ensemble[0].model = self.global_learners_ensemble[0].model.to(learner.device)
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)

                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters(), learner.device
                    )
                    
                '''for FedAvg-M or FedM'''
                if callable(getattr(learner.optimizer, "set_initial_grad", None)):
                    learner.optimizer.set_initial_grad(self.diff,learner.device)   
                    

                if callable(getattr(learner.optimizer, "set_initial_buffer", None)):

                    learner.optimizer.set_initial_buffer(self.buffer,learner.device)   
    
    def update_buffer_DOMO(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                if callable(getattr(learner.optimizer, "set_initial_buffer_DOMO", None)):
                    learner.optimizer.set_initial_buffer_DOMO(self.buffer, learner.device)  

    def mix_MGDA_Mom(self): #for MGDA_Mom

        self.sample_clients()

        print('local_E=',self.local_E)

        for client in self.sampled_clients:
            for i in range(self.local_E):
                client.step() # perform one step for the client

        if self.poisoned_flag:#先分配attck的client序号
            poisoned_clients=random.sample(range(0,len(self.sampled_clients)),self.poisoned_num)
            
        for learner_id, learner in enumerate(self.global_learners_ensemble): #learner are for FedEM
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            ''' w(t+1)=w(t)-eta(t)*d(t)
            eta(t-1)*g^i(t)=w0-w^i(t+1), d(t)=max(sum(lambda*g^i(t))), w0=w(t)
            Thus, w(t+1)=w(t)-eta(t)/eta(t-1)*max(sum(lambda*(w(t)-w^i(t+1))))
            if eta(t)=eta(t-1), we have w(t+1)=max(sum(lambda*w^i(t+1)))'''
            
            

            state_dict = learner.model.state_dict(keep_vars=True)
            cliped_value=copy.deepcopy(state_dict)
            # the parameter is set as the parameter upate     
                                                                                                         
            
            sampled_clients_copy=[client for client in self.sampled_clients]
            for client_id,sampled_client in enumerate(sampled_clients_copy):
                state_dict_initial = copy.deepcopy(state_dict )
                state_dict_current = sampled_client.learners_ensemble[learner_id].model.state_dict()
                for key in state_dict:
                    cliped_value[key].data=copy.deepcopy(state_dict_current[key].data).to(self.cuda_for_split)- state_dict_initial[key].data.clone()
                    sampled_clients_copy[client_id].learners_ensemble[learner_id].model.state_dict()[key].data=(
                        cliped_value[key].data.clone().to(sampled_client.learners_ensemble[learner_id].device))
                    

            #get w(t)
            global_parameter=[]
            for g_param in learner.model.parameters():
                if g_param.data is not None:
                    global_parameter.append(Variable(g_param.data.clone().to(learner.device), requires_grad=False))
            
            for p in learner.optimizer.param_groups:
                lr=p['lr']
            print('lr=',lr)
            
            '''get w^i(t+1)'''        
            local_parameter=[[] for i in range(len(self.sampled_clients))] #w^i(t+1)
            pn=[[] for i in range(len(self.sampled_clients))]
            param_differ=[[] for i in range(len(self.sampled_clients))] # w-w^i(t+1)
            for client_id, client in enumerate(sampled_clients_copy):
                clent_paramter=client.learners_ensemble[learner_id].model.parameters()
                for param in clent_paramter:
                    if param.data is not None:
                        local_parameter[client_id].append(Variable(param.data.clone().to(learner.device), requires_grad=False))
                
                #calculate w(t)-w^i(t+1)
                param_differ[client_id]=copy.deepcopy(local_parameter[client_id])
                
                # optional  
                pn[client_id] = parameter_normalizers(param_differ[client_id], 'l2') #normalization
                print('*************normalized*********')
                pn[client_id]=1
                
                #print('client_id',client_id,'pn_diff=',pn[client_id] )
                for pr_i in range(len(param_differ[client_id])):
                    param_differ[client_id][pr_i]=param_differ[client_id][pr_i]/pn[client_id]

            print('n_learners=',learner_id)
            print('start alpha calculation')
            '''calculate weight'''
            self.weights, min_norm = MinNormSolver.find_min_norm_element([param_differ[t] for t in range(len(self.sampled_clients))])
            self.weights=torch.tensor(self.weights)
            #weights, min_norm = MinNormSolver.find_min_norm_element([local_parameter[t] for t in range(len(self.sampled_clients))])
            eta=1
            print('alpha calculated')
            print('weights=',self.weights,'\n eta=',eta)
            #print('alpha=',weights)
            #average_learners(learners, learner, weights=weights) # defined in utils-> torch_utils

            self.diff,self.v_old=average_learners_FedMom(self.sampled_clients, learner, self.weights,self.v_old, eta, self.mu, pn)

        
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
           self.write_logs()
   
    #used for local_tune
    def local_tune(self):

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            for p in learner.optimizer.param_groups:
                p['lr']=0.001
                
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            for p in learner.optimizer.param_groups:
                print('local tune lr=',p['lr'])               
                
        for client in self.clients:
            for i in range(self.local_E):
                client.step()

        self.c_round += 1

        if self.c_round % 1 == 0:
            self.write_logs()        
            

class PersonalizedAggregator(CentralizedAggregator):
    r"""
    Clients do not synchronize there models, instead they only synchronize optimizers, when needed.

    """
    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(self.global_learners_ensemble[learner_id].model.parameters())


