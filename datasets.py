from utils import *

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter

matplotlib.rcParams.update({'font.size': 18})
eps = 1e-12


# Data Generation:
# Let's first generate random samples of rho, mu, D, and U. The true label is then calculated using the run_experiments function.
# The Reynolds numbers are also calculated and returned as a vector for convenience.
def sample_parameters_naive(num_samples=100000, Re_range=None, rho_range=[500, 3000], mu_range=[0.3e-3, 0.1], D_range=[0.05, 1], U_range=[1, 10], seed=123):
    if seed is not None:
        np.random.seed(seed)

    rho = np.random.uniform(rho_range[0], rho_range[1], num_samples)
    mu = np.random.uniform(mu_range[0], mu_range[1], num_samples)
    D = np.random.uniform(D_range[0], D_range[1], num_samples)
    U = np.random.uniform(U_range[0], U_range[1], num_samples)

    # Calculate the true CD and FD:
    CD, _ = run_experiments(Re_vec=None, velocities=U, diameters=D, densities=rho, viscosities=mu)
    Re_vec = rho * U * D / mu
    return rho, mu, D, U, Re_vec, CD

def sample_parameters(num_samples=100000,
                      Re_range=[1e2, 3e7],
                      rho_range=[500, 3000],
                      mu_range=[0.3e-3, 0.1],
                      D_range=[0.05, 1],
                      U_range=[1, 10],
                      seed=123):
    if seed is not None:
        np.random.seed(seed)
    
    rho = np.zeros(num_samples)
    mu = np.zeros(num_samples)
    D = np.zeros(num_samples)
    U = np.zeros(num_samples)
    
    # Step 1: Uniformly sample Re
    Re_vec = np.random.uniform(Re_range[0], Re_range[1], num_samples)
    
    # Parameters to choose from
    param_choices = ['rho', 'mu', 'D', 'U']
    
    for i in range(num_samples):
        # Step 2: Randomly choose which 3 parameters to sample
        sampled_params = np.random.choice(param_choices, 3, replace=False)
        
        # Step 3: Calculate the valid ranges for the sampled parameters
        if 'rho' in sampled_params:
            rho_min = max(rho_range[0], Re_vec[i] * mu_range[0] / (U_range[1] * D_range[1]))
            rho_max = min(rho_range[1], Re_vec[i] * mu_range[1] / (U_range[0] * D_range[0]))
            rho[i] = np.random.uniform(rho_min, rho_max)
        
        if 'mu' in sampled_params:
            mu_min = max(mu_range[0], rho_range[0] * U_range[0] * D_range[0] / Re_vec[i])
            mu_max = min(mu_range[1], rho_range[1] * U_range[1] * D_range[1] / Re_vec[i])
            mu[i] = np.random.uniform(mu_min, mu_max)
        
        if 'D' in sampled_params:
            D_min = max(D_range[0], Re_vec[i] * mu_range[0] / (rho_range[1] * U_range[1]))
            D_max = min(D_range[1], Re_vec[i] * mu_range[1] / (rho_range[0] * U_range[0]))
            D[i] = np.random.uniform(D_min, D_max)
        
        if 'U' in sampled_params:
            U_min = max(U_range[0], Re_vec[i] * mu_range[0] / (rho_range[1] * D_range[1]))
            U_max = min(U_range[1], Re_vec[i] * mu_range[1] / (rho_range[0] * D_range[0]))
            U[i] = np.random.uniform(U_min, U_max)
        
        # Step 4: Calculate the fourth parameter
        if 'rho' not in sampled_params:
            rho[i] = Re_vec[i] * mu[i] / (U[i] * D[i])
        elif 'mu' not in sampled_params:
            mu[i] = (rho[i] * U[i] * D[i]) / Re_vec[i]
        elif 'D' not in sampled_params:
            D[i] = Re_vec[i] * mu[i] / (rho[i] * U[i])
        elif 'U' not in sampled_params:
            U[i] = Re_vec[i] * mu[i] / (rho[i] * D[i])
            
    CD, _ = run_experiments(Re_vec=Re_vec, velocities=None, diameters=None, densities=None, viscosities=None)
    
    return rho, mu, D, U, Re_vec, CD


def sample_re(num_samples, Re_range, seed=None):
    if seed is not None:
        np.random.seed(seed)
    Re_vec = np.random.uniform(Re_range[0], Re_range[1], num_samples)
    CD, _ = run_experiments(Re_vec=Re_vec, velocities=None, diameters=None, densities=None, viscosities=None)
    return Re_vec, CD

# Used for verifying results later:


def extract_Re_values(dataset, subset):
    return [dataset.Re[idx] for idx in subset.indices]


class RandomDataset(Dataset):
    def __init__(self, num_samples=100000,
                 Re_range=[1e2, 3e7],
                 rho_range=[100, 2000],
                 mu_range=[0.001, 0.01],
                 D_range=[0.05, 0.5],
                 U_range=[0.1, 20],
                 seed=None):
        self.rho, self.mu, self.D, self.U, self.Re, self.CD = sample_parameters(
            num_samples=num_samples, Re_range=Re_range, rho_range=rho_range, mu_range=mu_range, D_range=D_range, U_range=U_range, seed=seed)
        # self.rho, self.mu, self.D, self.U, self.Re, self.CD = sample_parameters_naive(num_samples=num_samples, Re_range=None, rho_range=rho_range, mu_range=mu_range, D_range=D_range, U_range=U_range, seed=seed)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_sample = np.array([self.rho[idx], self.mu[idx], self.D[idx], self.U[idx]], dtype=np.float32)
        target = np.array([self.CD[idx]], dtype=np.float32)
        return input_sample, target


class RandomDataset_scaled(Dataset):
    def __init__(self, num_samples=100000,
                 parms_scaling=[1e3, 1e-2, 1, 1],  # divide rho, mu, D, U by these values
                 Re_range=[1e2, 3e7],
                 rho_range=[100, 2000],
                 mu_range=[0.001, 0.01],
                 D_range=[0.05, 0.5],
                 U_range=[0.1, 20],
                 seed=None):
        np.random.seed(seed)

        self.rho, self.mu, self.D, self.U, self.Re, self.CD = sample_parameters(num_samples=num_samples,
                                                                                Re_range=Re_range,
                                                                                rho_range=rho_range,
                                                                                mu_range=mu_range,
                                                                                D_range=D_range,
                                                                                U_range=U_range,
                                                                                seed=seed)
        
        # Check that the parameters combinations are consistent with Re values:
        Re_check = self.rho * self.U * self.D / self.mu
        assert np.allclose(Re_check, self.Re), "Re values are not consistent with the parameters"
        
        self.rho = self.rho / parms_scaling[0]
        self.mu = self.mu / parms_scaling[1]
        self.D = self.D / parms_scaling[2]
        self.U = self.U / parms_scaling[3]
        self.Re = self.Re / parms_scaling[0] / parms_scaling[2] / parms_scaling[3] * parms_scaling[1]

        # self.rho, self.mu, self.D, self.U, self.Re, self.CD = sample_parameters_naive(num_samples=num_samples, Re_range=None, rho_range=rho_range, mu_range=mu_range, D_range=D_range, U_range=U_range, seed=seed)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_sample = np.array([self.rho[idx], self.mu[idx], self.D[idx], self.U[idx]], dtype=np.float32)
        target = np.array([self.CD[idx]], dtype=np.float32)
        return input_sample, target


class RandomDataset_Re(Dataset):
    def __init__(self, num_samples, Re_range, re_scaling=1, seed=None):
        np.random.seed(seed)
        self.Re, self.CD = sample_re(num_samples=num_samples, Re_range=Re_range, seed=seed)
        self.num_samples = num_samples
        self.Re = self.Re / re_scaling

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_sample = np.array([self.Re[idx]], dtype=np.float32)
        target = np.array([self.CD[idx]], dtype=np.float32)
        return input_sample, target
