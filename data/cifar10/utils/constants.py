#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 20:13:14 2022

@author: user
"""

"""Configuration file for experiments"""
import string


LOADER_TYPE = {
    "synthetic": "tabular",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "emnist": "emnist",
    "imagenet":"imagenet",
    "emnist_alpha01": "emnist_alpha01",
    "femnist": "femnist",
    "shakespeare": "shakespeare",
}

EXTENSIONS = {
    "tabular": ".pkl",
    "cifar10": ".pkl",
    "cifar100": ".pkl",
    "imagenet":".pkl",
    "emnist": ".pkl",
    "emnist_alpha01": ".pkl",
    "femnist": ".pt",
    "shakespeare": ".txt",
}

AGGREGATOR_TYPE = {
    "FedMGDA": "centralized",
    "FedMom": "centralized",
    "FedNAG": "centralized",
    "FastSlowMo": "centralized",
    "DOMO": "centralized",
    'MIME': "centralized",
    "FedMoS":"centralized",
    "FedGLOMO": "centralized",
    "FedAdam":"centralized",
    "FedAGNL": "centralized",
    "FedMGDA-M-Mom":"centralized",
    "FedAvg-M-Mom":"centralized",
    "FedMGDA-Mom":"centralized",
    "FedAGNL-M": "centralized",
    "FedMGDA-tune":"centralized",
    "FedAvg-tune":"centralized",
    "FedMGDA-M": "centralized",
    "FedMGDA-M-DP": "centralized",
    "FedAvg+FedMGDA":"centralized",
    "FedProx+FedMGDA":"centralized",
    "FedM": "centralized",
    "FedM-DP": "centralized",
    "FedCM":"centralized",
    "FedSCM":"centralized",
    "FedEM": "centralized",
    "FedAvg": "centralized",
    "FedProx": "centralized",
    "Async_FedAvg":"centralized",
    "Async_FedAvg_DP":"centralized",
    "Async_FedMGDA":"centralized",
    "Async_FedMGDA_DP": "centralized",
    "Async_FedProx":"centralized",
    "FedAsync": "centralized",
    "local": "no_communication",
    "pFedMe": "personalized",
    "clustered": "clustered",
    "FeSEM":"EM",
    "APFL": "APFL",
    "L2SGD": "L2SGD",
    "AFL": "AFL",
    "FFL": "FFL"
}

CLIENT_TYPE = {
    "FedEM": "mixture",
    "AFL": "AFL",
    "FFL": "FFL",
    "APFL": "normal",
    "L2SGD": "normal",
    "FedAvg": "normal",
    "FedAvg+FedMGDA": "normal",
    "FedMGDA":"normal",
    "FedMom":"normal",
    "FedMoS":"normal",
    "FedGLOMO":"normal",
    "FedNAG":"normal",
    "FastSlowMo":"normal",
    "DOMO": "normal",
    "MIME": "normal",
    "FedAdam":"normal",
    "FedAGNL":"normal",
    "FedAGNL-M":"normal",
    "FedMGDA-Mom":"normal",
    "FedMGDA-M-Mom":"normal",
    "FedAvg-M-Mom":"normal",
    "FedMGDA-tune":"normal",
    "FedAvg-tune":"normal",
    "FedMGDA-M":"normal",
    "FedMGDA-M-DP":"normal",
    "FedProx+FedMGDA": "normal",
    "FedProx": "normal",
    "FedM": "normal",
    "FedM-DP": "normal",
    "FedCM":"normal",
    "FedSCM": "normal",
    "FeSEM":"normal",
    "Async_FedAvg":"normal",
    "Async_FedAvg_DP":"normal",
    "Async_FedMGDA":"normal",
    "Async_FedMGDA_DP":"normal",
    "Async_FedProx":"normal",
    "FedAsync":"normal",
    "local": "normal",
    "pFedMe": "normal",
    "clustered": "normal",
}

SHAKESPEARE_CONFIG = {
    "input_size": len(string.printable),
    "embed_size": 8,
    "hidden_size": 256,
    "output_size": len(string.printable),
    "n_layers": 2,
    "chunk_len": 80
}

CHARACTERS_WEIGHTS = {
    '\n': 0.43795308843799086,
    ' ': 0.042500849608091536,
    ',': 0.6559597911540539,
    '.': 0.6987226398690805,
    'I': 0.9777491725556848,
    'a': 0.2226022051965085,
    'c': 0.813311655455682,
    'd': 0.4071860494572223,
    'e': 0.13455606165058104,
    'f': 0.7908671114133974,
    'g': 0.9532922255751889,
    'h': 0.2496906467588955,
    'i': 0.27444893060347214,
    'l': 0.37296488139109546,
    'm': 0.569937324017103,
    'n': 0.2520734570378263,
    'o': 0.1934141300462555,
    'r': 0.26035705948768273,
    's': 0.2534775933879391,
    't': 0.1876471355731429,
    'u': 0.47430062920373184,
    'w': 0.7470615815733715,
    'y': 0.6388302610200002
}

