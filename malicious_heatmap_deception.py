from model import simulate_malicious
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

warnings.filterwarnings('ignore')


font_size = 16

# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.5

# number of pooled agents in each iteration
k = 10

# maximum iteration
max_iteration = 1000

# simulation times
simulation_times = 10

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.02

# percentage of malicious agents
malicious = 0.1

# confidence threshold
threshold= 0.5

# constant belief c
mal_c = 0.5

dampening = 0.001



file_name = 'malicious_heatmap/'
file_np = file_name + 'npy/'

pool_size = np.arange(14)+2
malicious_ls = np.linspace(0, 0.3, 16)
prob_evidence_ls = np.linspace(0, 0.1, 21)

# pull with deception
strategy = 'deception'
mal_c_ls = [None, 0.2, 0.5, 0.8]

for mal_c in mal_c_ls:
    # changes [evidence rate, proportion of malicious]
    avg_belief_mat = np.empty((prob_evidence_ls.size, malicious_ls.size))
    k = 5


    for i, prob_evidence in enumerate(np.flip(prob_evidence_ls)):
        for j, malicious in enumerate(malicious_ls):

            result = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                                        max_iteration=max_iteration, k=k, init_x = init_x,dampening=dampening,
                                        alpha=alpha, prob_evidence=prob_evidence,strategy=strategy,mal_c = mal_c,
                                        memory=False, memory_weight=0.5, malicious=malicious,threshold=threshold,
                                        pooling=True)

            avg_belief_mat[i,j] = result['belief_avg_true_good'].mean(axis=1)[-1]


    np.save(file_np + f'belief_avg_evidence_rate_malicious_{strategy}_mal_c_{mal_c}.npy', avg_belief_mat)
    plt.figure()
    sns.heatmap(pd.DataFrame(avg_belief_mat, index=np.flip(prob_evidence_ls), columns=malicious_ls), vmin=0, vmax=1)
    plt.title(f'pool size k = {k}')
    plt.ylabel('evidence rate', fontsize = font_size)
    plt.xlabel('number of malicious agents (percentage)', fontsize = font_size)
    plt.savefig(file_name + f'belief_avg_evidence_rate_malicious_{strategy}_mal_c_{mal_c}.png')


    # changes = [pool size, proportion of malicious]
    avg_belief_mat = np.empty((pool_size.size, malicious_ls.size))
    prob_evidence = 0.02


    for i, k in enumerate(np.flip(pool_size)):
        for j, malicious in enumerate(malicious_ls):

            result = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                                        max_iteration=max_iteration, k=k, init_x = init_x,dampening=dampening,
                                        alpha=alpha, prob_evidence=prob_evidence,strategy=strategy,mal_c = mal_c,
                                        memory=False, memory_weight=0.5, malicious=malicious,threshold=threshold,
                                        pooling=True)

            avg_belief_mat[i,j] = result['belief_avg_true_good'].mean(axis=1)[-1]

    np.save(file_np + f'belief_avg_k_malicious_{strategy}_mal_c_{mal_c}.npy', avg_belief_mat)
    plt.figure()
    sns.heatmap(pd.DataFrame(avg_belief_mat, index=np.flip(pool_size), columns=malicious_ls), vmin=0, vmax=1)

    plt.title(f'evidence rate = {prob_evidence}')
    plt.ylabel('pool size k', fontsize = font_size)
    plt.xlabel('number of malicious agents (percentage)', fontsize = font_size)
    plt.savefig(file_name + f'belief_avg_k_malicious_{strategy}_mal_c_{mal_c}.png')


    # changes = [pool size, evidence rate]
    avg_belief_mat = np.empty((pool_size.size, prob_evidence_ls.size))
    malicious = 0.2

    for i, k in enumerate(np.flip(pool_size)):
        for j, prob_evidence in enumerate(prob_evidence_ls):

            result = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                                        max_iteration=max_iteration, k=k, init_x = init_x,dampening=dampening,
                                        alpha=alpha, prob_evidence=prob_evidence,strategy=strategy,mal_c = mal_c,
                                        memory=False, memory_weight=0.5, malicious=malicious,threshold=threshold,
                                        pooling=True)



            avg_belief_mat[i,j] = result['belief_avg_true_good'].mean(axis=1)[-1]


    np.save(file_np + f'belief_avg_k_evidence_rate_{strategy}_mal_c_{mal_c}.npy', avg_belief_mat)
    plt.figure()
    sns.heatmap(pd.DataFrame(avg_belief_mat, index=np.flip(pool_size), columns=prob_evidence_ls), vmin=0, vmax=1)
    plt.title(f'proportion of malicious angents = {malicious}')
    plt.ylabel('pool size k', fontsize = font_size)
    plt.xlabel('evidence rate', fontsize = font_size)
    plt.savefig(file_name + f'belief_avg_k_evidence_rate_{strategy}_mal_c_{mal_c}.png')

