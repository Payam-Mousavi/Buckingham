
from utils import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np

matplotlib.rcParams.update({'font.size': 18})

BIN_SIZE = 500
SCALE_FACTOR = 100
SAMPLING_METHOD = "uniform-4"  # "uniform-Re" OR "uniform-4"

def update(frame, bins, bin_counts, scatter_dict, hist, ax, num_samples):
    rho_range = [200, 3000]
    mu_range = [0.6e-2, 1]
    D_range = [0.5, 1]
    U_range = [10, 20]
    
    if SAMPLING_METHOD == "uniform-4": 
        # Sample uniformly from 4 input parameters
        log_rho = np.random.uniform(np.log10(rho_range[0]), np.log10(rho_range[1]))
        log_U = np.random.uniform(np.log10(U_range[0]), np.log10(U_range[1]))
        log_D = np.random.uniform(np.log10(D_range[0]), np.log10(D_range[1]))
        log_mu = np.random.uniform(np.log10(mu_range[0]), np.log10(mu_range[1]))
        
        rho = np.power(10, log_rho)
        U = np.power(10, log_U)
        D = np.power(10, log_D)
        mu = np.power(10, log_mu)

        # Calculate Re and determine the corresponding bin
        Re = (rho * U * D) / mu
    elif SAMPLING_METHOD == "uniform-Re":
        # Sample uniformly from Re
        log_Re = np.random.uniform(3, 7)
        Re = np.power(10, log_Re)
    else: 
        raise ValueError("Invalid sampling method")

    bin_index = int((Re - 1e3) // BIN_SIZE)

    # Update the bin count and scatter plot
    if 0 <= bin_index < len(bin_counts):
        if SAMPLING_METHOD == "uniform-Re":
            if bin_counts[bin_index] == 0:
                bin_counts[bin_index] += 1
        else:
            bin_counts[bin_index] += 1

        CD_value = run_experiments(Re_vec=np.array([Re]))[0][0]
        if bin_index not in scatter_dict:
            scatter_dict[bin_index] = ax.scatter(bins[bin_index], CD_value, s=0, alpha=0.3)

        scatter_dict[bin_index].set_sizes(np.array([bin_counts[bin_index] * SCALE_FACTOR]))
    
    # Update the histogram
    hist.set_data(bins[:-1], bin_counts)
    ax_hist.set_ylim(0, max(bin_counts) * 1.1)  # Adjust the upper limit as needed

    # Update title with sampling efficiency
    repetitive_sampled_bins = bin_counts[bin_counts > 1]
    excess_samples = np.sum(repetitive_sampled_bins - 1)
    sampling_efficiency = 1 - (excess_samples / num_samples)
    ax.set_title(f"Sample Count: {frame + 1}/{num_samples}") # Sampling Efficiency: {sampling_efficiency * 100:.2f}%")
    
    return list(scatter_dict.values()) + [hist]

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    # plt.clf()
    ax = fig.add_subplot(211)
    ax.set_xlim(1e3, 1e7)  # Adjust the x-axis range as needed
    ax.set_ylim(0.1e-1, 6e-1)  # Adjust the y-axis range as needed
    ax.set_xscale('log')
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
    ax_hist.axhline(y=1.01, color='r', linestyle='--', linewidth=2)
    
    # Create the animation with combined update function
    num_samples = 200  # Total number of samples to be drawn
    animation = FuncAnimation(fig, update, frames=num_samples, fargs=(bins, bin_counts, scatter_dict, hist, ax, num_samples),
                              interval=50, repeat=False)
    
    file_path = f"Figures/sampling_{SAMPLING_METHOD}.mp4"
    animation.save(file_path, writer="ffmpeg", fps=10)
    
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()

     # Calculating and printing sampling efficiency:
    repetitive_sampled_bins = bin_counts[bin_counts > 1]
    excess_samples = np.sum(repetitive_sampled_bins - 1)
    print(f"Excess Samples: {excess_samples}")
    sampling_efficiency = 1 - (excess_samples / num_samples)
    print(f"Sampling Efficiency: {sampling_efficiency * 100:.2f}%")

