



from utils import *
from models import Model_II, Model_0
from datasets import *
from train import train_model
from predict import predict_model
from eval import eval_model

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

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


BIN_SIZE = 200
SCALE_FACTOR = 50

def update(frame, bins, bin_counts, scatter_dict, hist):
    rho_range = [200, 3000]
    mu_range = [0.6e-2, 1]
    D_range = [0.5, 1]
    U_range = [10, 20]
    
    # Sample from the input parameters
    rho = np.random.uniform(rho_range[0], rho_range[1])
    U = np.random.uniform(U_range[0], U_range[1])
    D = np.random.uniform(D_range[0], D_range[1])
    mu = np.random.uniform(mu_range[0], mu_range[1])
    
    # Calculate Re and determine the corresponding bin
    Re = (rho * U * D) / mu
    bin_index = int((Re - 1e3) // BIN_SIZE)
    
    # Update the bin count and scatter plot
    if 0 <= bin_index < len(bin_counts):
        bin_counts[bin_index] += 1
        CD_value = run_experiments(Re_vec=np.array([Re]))[0][0]
        if bin_index not in scatter_dict:
            scatter_dict[bin_index] = ax.scatter(bins[bin_index], CD_value, s=0, alpha=0.3)
        scatter_dict[bin_index].set_sizes(np.array([bin_counts[bin_index] * SCALE_FACTOR]))
    
    # Update the histogram
    hist.set_data(bins[:-1], bin_counts)
    ax_hist.set_ylim(0, max(bin_counts) * 1.1)  # Adjust the upper limit as needed
    ax.set_title(f"Sample Count: {frame + 1}/{num_samples}")
    
    return list(scatter_dict.values()) + [hist]

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    ax = fig.add_subplot(211)
    ax.set_xlim(1e3, 1e7)  # Adjust the x-axis range as needed
    ax.set_ylim(0.1e-1, 5e-1)  # Adjust the y-axis range as needed
    ax.set_xscale('log')
    # ax.set_xlabel('Re')
    ax.set_ylabel('CD')
    
    # Create bins and initialize bin counts
    bins = np.arange(1e3, 1e7 + BIN_SIZE, BIN_SIZE)
    bin_counts = np.zeros(len(bins) - 1)
    
    # Plot the CD curve
    Re_values = np.logspace(3, 7, 1000)
    CD_values, _ = run_experiments(Re_vec=Re_values)
    line, = ax.plot(Re_values, CD_values)
    
    # Create a dictionary to store scatter plots for each bin
    scatter_dict = {}
    
    # Set up the histogram subplot
    ax_hist = fig.add_subplot(212)
    ax_hist.set_xlim(1e3, 1e7)
    ax_hist.set_ylim(0, 1)  # Initial y-limit, will be updated later
    ax_hist.set_xscale('log')
    ax_hist.set_xlabel('Re')
    ax_hist.set_ylabel('Count')
    
    # Create the histogram plot using a line plot
    hist, = ax_hist.plot(bins[:-1], bin_counts, drawstyle='steps-post')
    
    # Create the animation
    num_samples = 100  # Total number of samples to be drawn
    animation = FuncAnimation(fig, update, frames=num_samples, fargs=(bins, bin_counts, scatter_dict, hist),
                              interval=100)
  
    animation.save("Figures/sampling_four-inputs.mp4", writer="ffmpeg", fps=10)

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()