import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader, random_split
# from torch.optim import Adam
# import torch.nn.init as init
# from torch.utils.tensorboard import SummaryWriter

matplotlib.rcParams.update({'font.size': 18})
eps = 1e-12

def CD_calc(Re=None, U=None, D=None, rho=None, mu=None):
    """
    Calculates the the Drag coefficient (CD) and the corresponding Drag force (FD), using the empirical equation found in the
    reference: https://pages.mtu.edu/~fmorriso/DataCorrelationForSphereDrag2016.pdf

    Use SI units

    """
    if Re is None:
        Re = rho*U*D/mu

    CD = (24/Re) \
        + ((2.6*(Re/5.)) / (1+(Re/5.)**1.52)) \
        + ((0.411*(Re/2.63e5)**-7.94) / (1 + (Re/2.63e5)**-8.)) \
        + ((0.25*(Re/1e6)) / (1 + (Re/1e6)))

    # Calculate FD if U and D and rho are given:
    if U is not None:
        FD = CD * (rho*np.pi/8) * (U**2) * (D**2)
    else:
        FD = None

    return CD, FD


def run_experiments(Re_vec, velocities=None, diameters=None, densities=None, viscosities=None):
    """
    Runs the experiment by iterating over the velocity vector or Re_vec and returning the drag coefficient CD and Drag force FD

    """
    if Re_vec is not None:
        CDs, FDs = CD_calc(Re=Re_vec, rho=None, mu=None, U=None, D=None)
    else:
        CDs, FDs = CD_calc(Re=None, U=velocities, D=diameters, rho=densities, mu=viscosities)

    return CDs, FDs


def plot_CD(Re, CD, marker='o'):
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=Re, y=CD, color='blue', marker=marker, alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$Re$')
    plt.ylabel('$C_D$')

    return True

def generate_inputs_from_Re(Re_vec, u_range, rho_range, mu_range, D_range):
    """
    Randomly sample from rho_range, mu_range, and D_range, calculate u, if u is in u_range, 
    output the parameters, u, rho, mu, D and if not, sample again and repeat. This is used to test inference.
    """
    # Generate random samples from the ranges
    size = len(Re_vec)
    rho = np.random.uniform(rho_range[0], rho_range[1], size)
    mu = np.random.uniform(mu_range[0], mu_range[1], size)
    D = np.random.uniform(D_range[0], D_range[1], size)
    U = Re_vec * mu / (rho * D)

    for u in U:
        if u >= u_range[1] and u <= u_range[0]:
            print(f"U = {u} is NOT in range")

    return rho, mu, D, U