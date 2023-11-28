#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:35:12 2023

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 20:12:59 2022

@author: user
"""

import warnings

import torch
import torch.nn as nn
import copy
import numpy as np
import random



def average_learners_FedAvg(
        learners,
        target_learner,
        weights,
        gamma_g,
        global_learner_id,
        ):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=target_learner.device)

    else:
        weights = weights.to(target_learner.device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0) 
    
    #for key in target_state_dict:                           
        #diff[key].data=(w0[key].data-target_state_dict[key].data)
        
    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[global_learner_id].model.state_dict(keep_vars=True)
                diff[key].data=w0[key].data- copy.deepcopy(state_dict[key].data).to(target_learner.device)
                target_state_dict[key].data.add_( diff[key].data, alpha=-gamma_g*weights[learner_id] )

    return diff


def average_learners_FedMom(
        learners,
        target_learner,
        weights,
        v_old,
        eta,
        beta,
        pn=None,
        ):
    """

        
    code for <<faster on-device training using new federated momentum algorithm>>'''
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=target_learner.device)

    else:
         weights = weights.to(target_learner.device)
            
            
    if pn is None:
        n_learners = len(learners)
        pn = 1 * torch.ones(n_learners, device=target_learner.device)
    print('pn=',pn)
    

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0) 
    avg_param=copy.deepcopy(w0) 
    v_new=copy.deepcopy(v_old) #v_t+1

    for key in target_state_dict:
        diff[key].data.fill_(0.)
        v_new[key].data.fill_(0.)
        avg_param[key].data.fill_(0.)
        
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[0].model.state_dict(keep_vars=True)
                avg_param[key].data+=weights[learner_id]*(w0[key].data - copy.deepcopy(state_dict[key].data).to(target_learner.device))/pn[learner_id]*max(pn)
    
    for key in target_state_dict:
          v_new[key].data=w0[key] - eta*avg_param[key].data
          target_state_dict[key].data=v_new[key].data + beta*(v_new[key].data-v_old[key].data)
    
    for key in target_state_dict:
        diff[key].data.copy_(w0[key].data-target_state_dict[key].data)
        
    return diff, v_new

def average_learners_FedNAG(
        learners,
        target_learner,
        weights,
        ):
    """

        
    code for <<faster on-device training using new federated momentum algorithm>>'''
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """
    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(target_learner.device)   
    
    avg_buffer=[]
    avg_momentum_buffer=copy.deepcopy(learners[0].learners_ensemble[0].optimizer.state_dict())
    for key in avg_momentum_buffer['state'].values():
        avg_buffer.append(key['momentum_buffer'].to(target_learner.device).mul_(0))
        
    ## aggregate the momentum term
    print('aggregate')
    for learner_id, learner in enumerate(learners):
        state_dict_op=learner.learners_ensemble[0].optimizer.state_dict()
 
        for i,state in enumerate(state_dict_op['state'].values()):
            avg_buffer[i].add_(copy.deepcopy(state['momentum_buffer']).to(target_learner.device),alpha=weights[learner_id])
    print('len avg',len(avg_buffer))  

       
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)
    
    # aggregate the model
    for key in target_state_dict:
        diff[key].data.fill_(0.)
        
        
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[0].model.state_dict(keep_vars=True)
                diff[key].data+=weights[learner_id]*copy.deepcopy(state_dict[key].data).to(target_learner.device)
    for key in target_state_dict:
        target_state_dict[key].data.copy_(diff[key].data)
        #print('valid')
        
    return avg_buffer




def average_learners_FastSlowMo(
        learners,
        target_learner,
        weights,
        global_buffer,
        beta,
        ):
    """

        
    code for <<FastSloWMo: Federated Learning with combined worker and aggregator momenta>>'''

    :beta, momentum coefficient
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor
    :global_buffer: global momentum buffer

    """
    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=target_learner.device)

    else:
        weights = weights.to(target_learner.device)   
    
    avg_buffer=[]
    avg_momentum_buffer=copy.deepcopy(learners[0].learners_ensemble[0].optimizer.state_dict())
    for key in avg_momentum_buffer['state'].values():
        avg_buffer.append(torch.zeros_like(key['momentum_buffer']).to(target_learner.device))
        
    ## aggregate the momentum
    print('aggregate')
    for learner_id, learner in enumerate(learners):
        state_dict_op=learner.learners_ensemble[0].optimizer.state_dict()
 
        for i,state in enumerate(state_dict_op['state'].values()):
            avg_buffer[i].add_(copy.deepcopy(state['momentum_buffer']).to(target_learner.device),alpha=weights[learner_id])

    # get the state dict of global model 
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    y = copy.deepcopy(w0)  #new buffer
    diff=copy.deepcopy(w0)
    
    # aggregate the model parameter
    for key in target_state_dict:
        y[key].data.fill_(0.)
        
    # get y(t)
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[0].model.state_dict(keep_vars=True)
                y[key].data+=weights[learner_id]*copy.deepcopy(state_dict[key].data).to(target_learner.device)
                
    # get x(t)
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
                        
            diff[key].data = y[key].data + beta*(y[key].data-global_buffer[key].data)
    
    # store y(t)
    global_buffer=copy.deepcopy(y)   

    #  load x(t) to the global model       
    for key in target_state_dict:
        target_state_dict[key].data.copy_(diff[key].data)
        #print('valid')
        
    return avg_buffer, global_buffer


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def average_learners_DOMO(
        learners,
        target_learner,
        weights,
        global_buffer,
        old_model, 
        K,
        lr,
        gamma_g,
        beta,
        mu
        ):
    """

        
    code for <<coordinating momenta for cross-silo federated learning>>'''
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    : lr, learning rate
    : K, local iteration
    : global_buffer: global momentum buffer
    :gamma_g: global learning rate
    : beta, global momentum coeffficient
    : old_model, model that is one step ahead
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=target_learner.device)

    else:
        weights = weights.to(target_learner.device)  
    
    # get the state dict of the global model 
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)



    # initialize old model
    if old_model is None:
        old_model=copy.deepcopy(w0)
        
    # get the d_t
    for key in diff:
        diff[key].data.fill_(0.)
        
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[0].model.state_dict(keep_vars=True)
                diff[key].data += (old_model[key].data-copy.deepcopy(state_dict[key].data).to(target_learner.device))*weights[learner_id]/(lr*K)
    
    
    #initialize the global buffer
    if global_buffer is None:
        global_buffer=copy.deepcopy(w0)
        for key in global_buffer:
            global_buffer[key].data.fill_(0.)
    
    # update the global buffer
    for key in global_buffer:
        if target_state_dict[key].data.dtype == torch.float32:
            global_buffer[key].data.mul_(beta).add_(diff[key].data)


        
    #update the global model:
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.add_(global_buffer[key].data, alpha=-gamma_g*lr*K)
            #premomentum fusion
            old_model[key].data =  target_state_dict[key].data - global_buffer[key].data*lr*mu*K
            
    
    
    
    return  global_buffer, old_model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def average_learners_MIME(
        learners,
        target_learner,
        weights,
        global_buffer,
        beta,
        ):
    """

        
    code for <<coordinating momenta for cross-silo federated learning>>'''
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    : global_buffer: global momentum buffer
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=target_learner.device)

    else:
        weights = weights.to(target_learner.device)   
    
    # get averaged gradient   
    gradient_buffer=[]
    sum_momentum_buffer=copy.deepcopy(learners[0].learners_ensemble[0].optimizer.state_dict())
    for idx, key in enumerate(sum_momentum_buffer['state'].values()):
        gradient_buffer.append(torch.zeros_like(key['gradient_buffer']).to(target_learner.device))

    for learner_id, learner in enumerate(learners):
        state_dict_op=learner.learners_ensemble[0].optimizer.state_dict()
        for idx,state in enumerate(state_dict_op['state'].values()):
            gradient_buffer[idx].add_(copy.deepcopy(state['gradient_buffer']).to(target_learner.device),alpha=weights[learner_id])

    # initialize the global model
    if global_buffer is None:
        print('+'*30)
        global_buffer=copy.deepcopy(gradient_buffer)
    
    # update the global model
    for idx,value in enumerate(gradient_buffer):
        global_buffer[idx].mul_(beta).add_(gradient_buffer[idx], alpha=1-beta)
            
    # get the state_dict of global model
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)
        
    for key in target_state_dict:
        diff[key].data.fill_(0.)
        
    #calculate the averaged model
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[0].model.state_dict(keep_vars=True)
                diff[key].data += copy.deepcopy(state_dict[key].data).to(target_learner.device)*weights[learner_id]
    
    #  load the averaged model to the global model       
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.copy_(diff[key].data)
        

    return  global_buffer

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def average_learners_FedMoS(
        learners,
        target_learner,
        weights,
        global_buffer,
        K,
        lr,
        beta,
        ):
    """

        
    code for <<coordinating momenta for cross-silo federated learning>>'''
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    : lr, learning rate
    : K, local iteration
    : global_buffer: global momentum buffer
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=target_learner.device)

    else:
        weights = weights.to(target_learner.device)   
    
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)

    #initilize the global momentum
    if global_buffer is None:
        global_buffer = copy.deepcopy(w0)
        for key in target_state_dict:
            global_buffer[key].data.fill_(0.)
    
    # calculate the difference between the global and local model
    for key in target_state_dict:
        diff[key].data.fill_(0.)
    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[0].model.state_dict(keep_vars=True)
                diff[key].data += (copy.deepcopy(state_dict[key].data).to(target_learner.device)-w0[key].data)*weights[learner_id]

    # update the global momumtuem
    for key in global_buffer:
        if target_state_dict[key].data.dtype == torch.float32:
            global_buffer[key].data.mul_(beta).add_(diff[key].data, alpha=-1.0/(lr*K))
                
    # update the global model    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.add_(global_buffer[key].data, alpha=-lr*K)
        

    return  global_buffer

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def average_learners_FedGLOMO(
        learners,
        target_learner,
        weights,
        beta,
        old_models_all,
        old_model,
        global_buffer
        ):
    """

        
    code for <<coordinating momenta for cross-silo federated learning>>'''
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    : lr, learning rate
    : K, local iteration
    : global_buffer: global momentum buffer
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(target_learner.device)   
    
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0_new=copy.deepcopy(target_state_dict)
    w0_old = copy.deepcopy(old_model)
    diff_new=copy.deepcopy(w0_new)
    diff_old=copy.deepcopy(old_model)

    # calculate the difference between the global and local model for old and new model
    for key in target_state_dict:
        diff_new[key].data.fill_(0.)
        diff_old[key].data.fill_(0.)
    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict_new = learner.learners_ensemble[0].state_dict(keep_vars=True)
                state_dict_old = old_models_all[learner_id]
                diff_new[key].data += (w0_new[key].data - copy.deepcopy(state_dict_new[key].data).to(target_learner.device))*weights[learner_id]
                diff_old[key].data += (w0_old[key].data - state_dict_old[key].data)*weights[learner_id]
    
    
    # update the global momumtuem
    if global_buffer is None:
        global_buffer = copy.deepcopy(diff_new)
        
    else:
        for key in global_buffer:
            if global_buffer[key].data.dtype == torch.float32:
                global_buffer[key].data.mul_(1-beta).add_(diff_new[key].data, alpha=beta)
                global_buffer[key].data.add_(diff_new[key].data-diff_old[key].data, alpha=1-beta)

    #update the old model
    old_model = copy.deepcopy(target_state_dict)
    
    # update the global model    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.add_(global_buffer[key].data, alpha=-1)
        

    return  old_model, global_buffer
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def average_learners_FedAdam(
        learners,
        target_learner,
        weights,
        m_old,
        v_old,
        mu,
        beta,
        gamma_g
        ):
    """

    
    code for <<faster on-device training using new federated momentum algorithm>>'''
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """
    print('Fedadam aggregation','\n mu=',mu, ' \gamma_g=',gamma_g, ' beta=',beta)
    eps=1e-8
    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=target_learner.device)

    else:
        weights = weights.to(target_learner.device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0) 
    mt=copy.deepcopy(w0)
    vt=copy.deepcopy(w0)


    for key in target_state_dict:
        diff[key].data.fill_(0.)
        mt[key].data.fill_(0.)
        vt[key].data.fill_(0.)
        
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.learners_ensemble[0].model.state_dict(keep_vars=True)
                diff[key].data+=weights[learner_id]*(copy.deepcopy(state_dict[key].data).to(target_learner.device)-w0[key].data)
    for key in target_state_dict:
        mt[key].data=mu*m_old[key].data + (1-mu)*diff[key].data
        vt[key].data=beta*v_old[key].data + (1-beta)*diff[key].data.mul(diff[key].data)
        denorm=torch.sqrt(vt[key].data)+eps
        # print('demore,=',denorm)
        target_state_dict[key].data=w0[key].data + gamma_g*torch.div(mt[key].data,denorm)
        
    return mt, vt

#%%%
def average_learners_FedAGNL(
        learners,
        target_learner,
        weights,
        w_old,
        alpha,
        beta,
        gamma_g,
        mu
        ):
    """

    
    :alpha
    :beta corresponding to theta n
    :gamma_g coreesponingd to  kn
    : mu=1
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """
    print('FedAGNL aggregation','\n alpha=',alpha,  ' beta=',beta,' \gamma_g=',gamma_g,'mu=',mu)
    eps=1e-8
    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0) 
    delt=copy.deepcopy(w0)
    


    for key in target_state_dict:
        diff[key].data.fill_(0.)
        delt[key].data.fill_(0.)

        
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)
                diff[key].data+=weights[learner_id]*(state_dict[key].data-w0[key].data)
    #no line search
    
    for key in target_state_dict:
        delt[key].data=alpha*(w0[key].data-w_old[key].data)+beta*diff[key].data/torch.norm(diff[key].data.float())+gamma_g*diff[key].data
        target_state_dict[key].data.copy_(w0[key].data + mu*delt[key].data)
        
    for key in target_state_dict:
        diff[key].data.copy_(w0[key].data-target_state_dict[key].data)
    
    return w0, diff
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def average_learners_MGDA(
        learners,
        target_learner,
        weights,
        gamma_g=1):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """


    # if weights is None:
    #     n_learners = len(learners)
    #     weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    # else:
    #     weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)
                target_state_dict[key].data += gamma_g*weights[learner_id] * state_dict[key].data.clone()

    for key in target_state_dict:                           
        diff[key].data=(w0[key].data-target_state_dict[key].data)
    
    return diff

def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    # else:
    #     weights = weights.to(learners[0].device)
    
    
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    
    diff=copy.deepcopy(w0)
    
    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    #print('key=',key)
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()
    for key in target_state_dict:                           
        diff[key].data=(w0[key].data-target_state_dict[key].data)
    
    return diff






def partial_average(learners, average_learner, alpha):
    """
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data =\
                    (1-alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data


def differentiate_learner(target, reference_state_dict, coeff=1.):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            target_state_dict[key].grad = \
                coeff * (target_state_dict[key].data.clone() - reference_state_dict[key].data.clone())


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def simplex_projection(v, s=1):
    """
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    .. math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = - float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w


