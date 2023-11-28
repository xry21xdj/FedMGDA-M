import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np
import copy


class ProxSGD(Optimizer):
    r"""Adaptation of  torch.optim.SGD to proximal stochastic gradient descent (optionally with momentum),
     presented in `Federated optimization in heterogeneous networks`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Attributes
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = ProxSGD(model.parameters(), lr=0.1, mu=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, mu=0., momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ProxSGD, self).__init__(params, defaults)

        self.mu = mu

        for group in self.param_groups:
            for p in group['params']:
                #print('p=',p)
                param_state = self.state[p]
                param_state['initial_params'] = torch.clone(p.data)


    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        count=0
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #count+=1
                #print('count=',count)
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # add proximal term
                d_p.add_(p.data - param_state['initial_params'], alpha=self.mu)
                #add_代表+, 里面括号代表乘
                p.data.add_(d_p, alpha=-group['lr'])

        return loss

    def set_initial_params(self, initial_params,device):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups = list(initial_params)
        if len(initial_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(initial_param_groups[0], dict):
            initial_param_groups = [{'params': initial_param_groups}]

        for param_group, initial_param_group in zip(self.param_groups, initial_param_groups):
            for param, initial_param in zip(param_group['params'], initial_param_group['params']):
                param_state = self.state[param]
                param_state['initial_params'] = torch.clone(initial_param.data).to(device)


'''FedNGA optimizer: NAG'''

class NAG_SGD(Optimizer):
    r""" optimizer with nesterove
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = NAG_SGD(model.parameters(), lr=0.1, mu=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, gamma=0., dampening=0.,
                 weight_decay=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,  dampening=dampening, gamma=gamma,
                        weight_decay=weight_decay)
        self.momentum_buffer={}
        super(NAG_SGD, self).__init__(params, defaults)

        self.gamma = gamma

        # for group in self.param_groups:
        #     for p in group['params']:
        #         #print('p=',p)
        #         param_state = self.state[p]
        #         param_state['initial_params'] = torch.clone(p.data)
           
                

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        count=0
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            gamma = group['gamma']
            dampening = group['dampening']
            lr=group['lr']


            for p in group['params']:
                
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                    # if(count==0):
                    #     print('buf=',torch.norm(buf))
                    #     count+=1
                    buf.mul_(gamma).add_(d_p, alpha=-lr) # gamma*v-lr*d_p

                    param_state['momentum_buffer'].copy_(buf) 
                p.data.add_(d_p, alpha=-lr)
                p.data.add_(buf, alpha=gamma)
        return loss

    def set_initial_buffer(self, initial_buffer, device):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        #print('sending average momentum')
        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                #print('i=',i)
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer']=torch.zeros_like(p.data)
                self.state[p]['momentum_buffer'].copy_(initial_buffer[i].to(device))
        # self.load_state_dict(self.state_dict())


# optimizer for FastSlowMo
class FastSlowMo_SGD(Optimizer):
    r""" optimizer for local side of FastSlowMO algorithm
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    gamma (float): momentum coefficient for Fast momentum
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)



    math:
        y(t)=x(t-1) - lr* \nabla f(x(t-1))
        x(t)=y(t) + gamma*(y(t)-y(t-1))
    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = FastSlowMo_SGD(model.parameters(), lr=0.1, gamma=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, gamma=0., dampening=0.,
                 weight_decay=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,  dampening=dampening, gamma=gamma,
                        weight_decay=weight_decay)
        self.momentum_buffer={}
        super(FastSlowMo_SGD, self).__init__(params, defaults)

        self.gamma = gamma


                

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        count=0
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            gamma = group['gamma']
            dampening = group['dampening']
            lr=group['lr']


            for p in group['params']:
                
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = torch.zeros_like(p.data)
                    param_state['momentum_buffer'].copy_(p.data)
                    
                buf = p.data-d_p.mul(lr)
                p.data.add_(d_p, alpha=-lr)
                p.data.add_(buf-param_state['momentum_buffer'], alpha=gamma)
                param_state['momentum_buffer'].copy_(buf) 
                
        return loss

    def set_initial_buffer(self, initial_buffer,device):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        #print('sending average momentum')
        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                #print('i=',i)
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer']=torch.zeros_like(p.data)
                self.state[p]['momentum_buffer'].copy_(initial_buffer[i].to(device))
        # self.load_state_dict(self.state_dict())




# optimizer for DOMO
class DOMO_SGD(Optimizer):
    r""" optimizer for local side of DOMO algorithm
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    gamma (float): momentum coefficient for Fast momentum
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)



    math:
        x(t)=x(t-1) - eta*beta*K*y
        y(t)=mu*y(t-1) + \nabla(f(x(t)))
        
        x(t+1) = x(t)-eta*y(t)
        
    lr: copresponding to eta,
    gamma: corresponding to alpha
    mu: corresponding to mu in the paper
    
    
    
    Methods
    ----------
    __init__
    __step__
    set_initial_buffer

    Example
    ----------
        >>> optimizer = DOMO(model.parameters(), lr=0.1, gamma=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, gamma=0., mu=0,  K=1,
                 weight_decay=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,  gamma=gamma, mu=mu,K=K,
                        weight_decay=weight_decay)

        super(DOMO_SGD, self).__init__(params, defaults)


                

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        count=0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            #gamma = group['gamma']

            lr=group['lr']
            mu = group['mu']
            K=group['K']
            #print('k=',K)
            
            for p in group['params']:

                if p.grad is None:
                    print('grad==none*****')
                    continue
                d_p = p.grad.data

                # if weight_decay != 0:
                #     d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                #print('state=',param_state)
                if 'momentum_buffer' not in param_state:
                    print('++++++first time*****')
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    # param_state['sum_momentum_buffer'] = torch.zeros_like(p.data)
                    
                    # param_state['origin_buffer'] = copy.deepcopy(p.data)
                else:
                    local_buf = copy.deepcopy(param_state['momentum_buffer'])
                    # update local momentum
                    
                    local_buf.mul_(mu).add_(d_p)
                    
                    #update the local moemtum buffer
                    param_state['momentum_buffer'].copy_(local_buf) 
                

                    
                    d_p.copy_(local_buf)

                
                p.data.add_(d_p, alpha=-lr)
                
                
        return loss

    def set_initial_buffer_DOMO(self, initial_buffer,device):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups=[] 
        for keys in initial_buffer:
            if 'weight' in keys:
                initial_param_groups.append(initial_buffer[keys].to(device))
            if 'bias' in keys:
                initial_param_groups.append(initial_buffer[keys].to(device))
                
        #print('sending momentum *****')
        for group in self.param_groups:
            for i,p in enumerate(group['params']):

                #update the first local model with gloabl momentum
                p.data.add_(initial_param_groups[i], alpha=-group['lr']*group['mu']*group['K'])
                
                
                # set the local momentum buffer to zero every communication round
                self.state[p]['momentum_buffer']=torch.zeros_like(p.data)
                #self.state[p]['sum_momentum_buffer']=torch.zeros_like(p.data)
                


# optimizer for MIME
class MIME_SGD(Optimizer):
    r""" optimizer for local side of MIME algorithm
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    gamma (float): momentum coefficient for Fast momentum
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)



    math:
       
        w(t)^{i,k}=w(t)^{i,k+1}-lr*((1-beta)\nabla f(w(t)^{i,k} + beta u(t)))
        
    lr: copresponding to eta,
    beta: corresponding to beta
    
    
    
    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = MIME_SGD(model.parameters(), lr=0.1,beta=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required,  beta=0, K=1,
                 weight_decay=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,  beta=beta, K=K,
                        weight_decay=weight_decay)

        super(MIME_SGD, self).__init__(params, defaults)


                

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        count=0
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']

            lr=group['lr']
            beta = group['beta']
            K= group['beta']
            i=0
            for p in group['params']:
                i+=1
                #print('p=',p)
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                #print('state=',param_state)
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    param_state['momentum_buffer'].copy_(d_p)

                    
                    
                local_buf = d_p.mul(1-beta) + param_state['momentum_buffer'].mul(beta)
                
                p.data.add_(local_buf, alpha=-lr)
            #print('P=',i)
        return loss

    def set_initial_buffer(self, initial_buffer,device):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        #print('sending momentum *****')
        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                #print('p=',p)
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer']=torch.zeros_like(p.data)
                #update the mometum buffer    
                self.state[p]['momentum_buffer']=initial_buffer[i].to(device)
                
    def set_full_gradient_MIME(self):
        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                self.state[p]['gradient_buffer'] = p.grad.data
                
                
# optimizer for FedMos
class FedMoS_SGD(Optimizer):
    r""" optimizer for local side of FedMoS algorithm
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    gamma (float): momentum coefficient for Fast momentum
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)



    math:
        u(t)^{i,k+1}=\nabla f_i(w(t)^{i,k}) + (1-gamma)*(u(t)^{i,k}- \nabla f_i(w(t)^{i,k-1}))
        w(t)^{i,k+1} = w(t)^{i,k+1} - lr*u(t)^{i,k+1} - mu*(w(t)^{i,k} - w(t))
        
    
    
    Methods
    ----------
    __init__
    __step__
    set_initial_buffer

    Example
    ----------
        >>> optimizer = FedMoS_SGD(model.parameters(), lr=0.1, gamma=0.9,mu=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, gamma=0., mu=0,  K=1,
                 weight_decay=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,  gamma=gamma, mu=mu,K=K,
                        weight_decay=weight_decay)

        super(FedMoS_SGD, self).__init__(params, defaults)
        
        # a count to decide the first local step
        self.count=0
        
        #totoal local steps:
        self.K =K
                

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
    
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            gamma = group['gamma']

            lr=group['lr']
            mu = group['mu']
            K=group['K']

            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                
                #to find the first local step
                if self.count==0:
                    #print('++++++first step*****')
                    param_state['gradient_buffer'] = torch.zeros_like(p.data)
                    param_state['model_buffer'] = copy.deepcopy(p.data)
                    
                    
                # update the momentum buffer
                else:
                    
                    local_buf = copy.deepcopy(param_state['momentum_buffer'])
                    local_buf.add_(param_state['gradient_buffer'], alpha=-1)
                    local_buf.mul_(gamma).add_(d_p)
                    param_state['momentum_buffer'].copy_(local_buf) 
                    
                # update the model
                p.data.mul_(1-mu).add_(param_state['momentum_buffer'], alpha=-lr)
                p.data.add_(param_state['model_buffer'], alpha=mu)
                
                # copy the old gradient
                param_state['gradient_buffer'] = copy.deepcopy(d_p)
                
                
        # reset the count to get the first local step
        self.count+=1
        if self.count==self.K:
            self.count=0
        
        
        
        return loss

    def set_initial_buffer_FedMoS(self):
        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                self.state[p]['momentum_buffer'] = p.grad.data
                
'''FedGLOMO optimizer: FedGLOMO'''

class FedGLOMO_SGD(Optimizer):
    r""" optimizer with nesterove
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = FedGLOMO_SGD(model.parameters(), lr=0.1, mu=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, K=1,
                 weight_decay=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, K=K,
                        weight_decay=weight_decay)
        self.momentum_buffer={}
        super(FedGLOMO_SGD, self).__init__(params, defaults)

        self.K=K
        self.count = 0 #count to locate the first local iteration
                

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr=group['lr']


            for p in group['params']:
                
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                
                if self.count ==0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    param_state['momentum_buffer'].copy_(d_p)
                    param_state['gradient_buffer'] = copy.deepcopy(d_p)
                    
                else:
                    # update local momentum
                    local_buf = param_state['momentum_buffer']
                    local_buf.add_(d_p-param_state['gradient_buffer']) 
                    param_state['momentum_buffer'].copy_(local_buf) 
                    # record the last step's gradient
                    param_state['gradient_buffer'] = copy.deepcopy(d_p)
                    
                # update the model
                p.data.add_(param_state['momentum_buffer'], alpha=-lr)
            
            # reset count if all local iteration finished
            self.count +=1
            if self.count == int(self.K):
                self.count=0
                
        return loss


class FedCM_SGD(Optimizer):
    r"""Adaptation of  torch.optim.SGD to proximal stochastic gradient descent (optionally with momentum),
     presented in `Federated optimization in heterogeneous networks`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    the code is implemented based on the paper: FedCM: Federated Learning with
    Client-level Momentum,  https://arxiv.org/pdf/2106.10874.pdf
    
    \math
    
    v(t)=beta*delta(t)+(1-beta)*g(t)
    w(t+1)=w(t)-lr*v(t)
    
    Attributes
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    beta (float, optional): parameter for client level momentum
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = FedCM_SGD(model.parameters(), lr=0.1, beta=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, beta=0., K=1, momentum=0., dampening=0.,
                 weight_decay=0., agg_round=1, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedCM_SGD, self).__init__(params, defaults)

        self.beta = beta
        self.K=K
        self.agg_round=agg_round
        self.count=1
        for group in self.param_groups:
            self.lr=group['lr']
            self.delta=[[] for i in range(len(group['params']))]
                
                #self.delta['initial_params']=p.grad


    def __setstate__(self, state):
        super(FedCM_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.agg_round==1:
            for group in self.param_groups:
                for idx, p in enumerate(group['params']):
                    #param_state['initial_params'] = p.grad.data
                    self.delta[idx]=p.grad.data
                    #print('len_delta',len(self.delta))
                #print ('count=',self.count)
            self.agg_round+=1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print('len_dp=',len(d_p))
                if self.delta[idx]==[]:
                    print('self.delta_initial')
                    self.delta[idx]=torch.clone(d_p).detach()
                else:
                    dp=self.delta[idx].mul_(self.beta).add_(d_p, alpha=1-self.beta)

                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf


                #add_代表+, 里面括号代表乘
                p.data.add_(d_p, alpha=-group['lr'])

        return loss

            
    def set_initial_grad(self, initial_params,device):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups=[] 
        for keys in initial_params:
            if 'weight' in keys:
                initial_param_groups.append(initial_params[keys].to(device))
            if 'bias' in keys:
                initial_param_groups.append(initial_params[keys].to(device))
        if len(initial_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        #print('len_init_parm=',len(initial_param_groups))
        #print('len_delta=',len(self.delta),'len_init',len(initial_param_groups))
        for idx, grad in enumerate(initial_param_groups):
            self.delta[idx]=grad/self.lr/self.K
    
class FedSCM_SGD(Optimizer):
    r"""Adaptation of  torch.optim.SGD to proximal stochastic gradient descent (optionally with momentum),
     presented in `Federated optimization in heterogeneous networks`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    
    \math
    
    v(t+1)=beta*v(t)+(1-beta)*g(t)
    w(t+1)=w(t)-lr*v(t+1)
    
    Attributes
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    beta (float, optional): parameter for client level momentum
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = FedCM_SGD(model.parameters(), lr=0.1, beta=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, beta=0., K=1, momentum=0., dampening=0.,
                 weight_decay=0., agg_round=1, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedSCM_SGD, self).__init__(params, defaults)

        self.beta = beta
        self.K=K
        self.agg_round=agg_round
        self.count=1
        self.S_beta=self.K-self.beta*(1-pow(self.beta, self.K))/(1-self.beta)
        for group in self.param_groups:
            self.delta=[[] for i in range(len(group['params']))]
            self.lr=group['lr']
                
                #self.delta['initial_params']=p.grad


    def __setstate__(self, state):
        super(FedSCM_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print('len_dp=',len(d_p))

                
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                    
                if self.delta[idx]==[]:
                    print('self.delta_initial for scm_sgd')
                    self.delta[idx]=torch.clone(d_p).detach()
                else:
                    self.delta[idx].mul_(self.beta).add_(d_p, alpha=1-self.beta)
                d_p = self.delta[idx]
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf


                #add_代表+, 里面括号代表乘
                p.data.add_(d_p, alpha=-group['lr'])

        return loss

    # def set_initial_params(self, initial_params):
    #     r""".
    #         .. warning::
    #             Parameters need to be specified as collections that have a deterministic
    #             ordering that is consistent between runs. Examples of objects that don't
    #             satisfy those properties are sets and iterators over values of dictionaries.

    #         Arguments:
    #             initial_params (iterable): an iterable of :class:`torch.Tensor` s or
    #                 :class:`dict` s.
    #     """
    #     initial_param_groups = list(initial_params)
    #     if len(initial_param_groups) == 0:
    #         raise ValueError("optimizer got an empty parameter list")
    #     #print('len_init_parm=',len(initial_param_groups))
    #     for idx, p in enumerate(initial_param_groups):
    #         self.delta[idx]=p.grad.data
            
    def set_initial_grad(self, initial_params,local_E):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups=[] 
        for keys in initial_params:
            if 'weight' in keys:
                initial_param_groups.append(initial_params[keys])
            if 'bias' in keys:
                initial_param_groups.append(initial_params[keys])
        if len(initial_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        #print('len_init_parm=',len(initial_param_groups))
        #print('len_delta=',len(self.delta),'len_init',len(initial_param_groups))
        for idx, grad in enumerate(initial_param_groups):
            self.delta[idx]=grad/self.lr/self.K/local_E
            
class FedM_SGD(Optimizer):
    r"""Adaptation of  torch.optim.SGD to proximal stochastic gradient descent (optionally with momentum),
     presented in `Federated optimization in heterogeneous networks`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    
    \math
    
    v(t+1)=beta*v(t)+g(t)
    w(t+1)=w(t)-lr*v(t+1)
    
    Attributes
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    beta (float, optional): parameter for client level momentum
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = FedCM_SGD(model.parameters(), lr=0.1, beta=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, beta=0., K=1, momentum=0., dampening=0.,
                 weight_decay=0., agg_round=1, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedM_SGD, self).__init__(params, defaults)

        self.beta = beta
        self.K=K
        self.agg_round=agg_round
        self.count=1
        self.S_beta=((1-self.beta)*self.K-self.beta*(1-pow(self.beta, self.K)))/(pow((1-self.beta),2))
        print('s_beta=',self.S_beta,'beta=',self.beta,'K=',self.K)
        for group in self.param_groups:
            self.delta=[[] for i in range(len(group['params']))]
            self.lr=group['lr']
                
                #self.delta['initial_params']=p.grad


    def __setstate__(self, state):
        super(FedM_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """


        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print('len_dp=',len(d_p))
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                    
                if self.delta[idx]==[]:
                   # print('self.delta_initial')
                    print('s_beta=',self.S_beta)
                    self.delta[idx]=torch.clone(d_p).detach()
                else:
                    self.delta[idx].mul_(self.beta).add_(d_p, alpha=1)
                d_p = self.delta[idx]
                


                #add_代表+, 里面括号代表乘
                p.data.add_(d_p, alpha=-group['lr'])

        return loss

    
    def set_initial_grad(self, initial_params,device):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups=[] 
        for keys in initial_params:
            if 'weight' in keys:
                initial_param_groups.append(initial_params[keys].to(device))
            if 'bias' in keys:
                initial_param_groups.append(initial_params[keys].to(device))
        if len(initial_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        #print('len_init_parm=',len(initial_param_groups))
        #print('len_delta=',len(self.delta),'len_init',len(initial_param_groups))
        for idx, grad in enumerate(initial_param_groups):
            self.delta[idx]=grad/self.lr/self.S_beta

def get_optimizer(optimizer_name, model, lr_initial, mu=0.,gamma=0., beta=0.,K=1):
    """
    Gets torch.optim.Optimizer given an optimizer name, a model and learning rate

    :param optimizer_name: possible are adam and sgd
    :type optimizer_name: str
    :param model: model to be optimized
    :type optimizer_name: nn.Module
    :param lr_initial: initial learning used to build the optimizer
    :type lr_initial: float
    :param mu: proximal term weight; default=0.
    :type mu: float
    :return: torch.optim.Optimizer

    """

    if optimizer_name == "adam":
        return optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            weight_decay=5e-4
        )

    elif optimizer_name == "sgd":
        print('optimizer=','sgd')
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4
        )

    elif optimizer_name == "prox_sgd":
        print('optimizer=','prox_sgd')
        return ProxSGD(
            [param for param in model.parameters() if param.requires_grad],
            mu=mu,
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4
        )
    elif optimizer_name == "cm_sgd":
        print('optimizer=','cm_sgd')
        return FedCM_SGD(
            [param for param in model.parameters() if param.requires_grad],
            beta=beta,
            K=K,
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4
        )
    elif optimizer_name == "scm_sgd":
        print('optimizer=','scm_sgd')
        return FedSCM_SGD(
            [param for param in model.parameters() if param.requires_grad],
            beta=beta,
            K=K,
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4
        )
    elif optimizer_name == "m_sgd":
        print('optimizer=','m_sgd')
        return FedM_SGD(
            [param for param in model.parameters() if param.requires_grad],
            beta=beta,
            K=K,
            lr=lr_initial,
            momentum=0,
            weight_decay=5e-4
        )
    elif optimizer_name == "nag_sgd":
        print('optimizer=','nag_sgd')
        return NAG_SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            gamma=gamma,
            weight_decay=5e-4
        )
    elif optimizer_name == "fastslowmo_sgd":
        print('optimizer=','fastslowmo_sgd')
        return FastSlowMo_SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            gamma=gamma,
            weight_decay=5e-4
        )
    elif optimizer_name == "domo_sgd":
        print('optimizer=','domo_sgd')
        return DOMO_SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            gamma=gamma,
            mu=mu,
            K=K,
            weight_decay=5e-4
        )
    elif optimizer_name == "mime_sgd":
        print('optimizer=','mime_sgd')
        return MIME_SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            beta=beta,
            K=K,
            weight_decay=5e-4
        )
    elif optimizer_name == "fedmos_sgd":
        print('optimizer=','fedmos_sgd')
        return FedMoS_SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            gamma=gamma,
            mu=mu,
            K=K,
            weight_decay=5e-4
        )
    elif optimizer_name == "fedglomo_sgd":
        print('optimizer=','fedglomo_sgd')
        return FedGLOMO_SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            K=K,
            weight_decay=5e-4
        )
    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")

