import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import simulate_malfunctioning
import warnings

warnings.filterwarnings('ignore')
font_size = 16

# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.4

# number of pooled agents in each iteration
k = 5

# maximum iteration
max_iteration = 2000

# simulation times
simulation_times = 5

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.02

# percentage of malfunctioning agents
malfunctioning = 0.20

# threshold for fault detection
threshold = 0.5

# weights_updating
weights_updating = 'kl_divergence'


file_name = 'equal_weights/'

dampening = None

pool_size = [10, 12, 15]


for k in pool_size:

    result = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n,
                      max_iteration=max_iteration, k=k, init_x = init_x, weights_updating=weights_updating,dampening=dampening,
                      mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                      memory=False, memory_weight=0.5, malfunctioning=malfunctioning, threshold=threshold,
                      pooling=True)

    acc = result['accuracy_avg'].mean(axis=1)
    avg_belief = result['belief_avg_true_good'].mean(axis=1)

    result_ew = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n, equal_weights=True,
                                     max_iteration=max_iteration, k=k, init_x = init_x, weights_updating=weights_updating,dampening=dampening,
                                     mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                                     memory=False, memory_weight=0.5, malfunctioning=malfunctioning, threshold=threshold,
                                     pooling=True)

    acc_ew = result_ew['accuracy_avg'].mean(axis=1)
    avg_belief_ew = result_ew['belief_avg_true_good'].mean(axis=1)

    plt.figure(f'acc k {k}')
    plt.plot(range(max_iteration), acc)
    plt.plot(range(max_iteration), acc_ew)
    plt.legend(['different weights', 'equal weights'])
    plt.xlabel('iteration')
    plt.ylabel('time for detection')
    plt.savefig(file_name + f'equal_weights_detection_k_{k}.png')

    plt.figure(f'detection k {k}')
    plt.plot(range(max_iteration), avg_belief)
    plt.plot(range(max_iteration), avg_belief_ew)
    plt.legend(['different weights', 'equal weights'])
    plt.xlabel('iteration')
    plt.ylabel('average belief of functioning agents')
    plt.savefig(file_name + f'equal_weights_avg_belief_k_{k}')


