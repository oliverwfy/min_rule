from model import simulate_malicious_acc, simulate_malicious_acc_bc
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pandas as pd


import numpy as np

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
max_iteration = 10

# simulation times
simulation_times = 1

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.01
weights_updating = 'kl_divergence'


# percentage of malicious agents
malicious = 0.1


file_name = 'malicious_acc/'


threshold= 0.5
k = 10
mal_x = 0.5

mal_c = 0.5

noise = 0.2
# dampening
dampening = 0.001

strategy = 'deception'

malicious_ls = [0, 0.05, 0.1, 0.15, 0.2]

mal_percentage = np.arange(0,31,5) / 100.0

pool_size = [20, 15, 10, 6, 3, 2, 1]
mal_x_ls = np.round(np.linspace(0.05, 0.6, 12), 2)

# change : ["pool size", "mal_percentage"]
avg_belief_mat = np.zeros((len(pool_size), len(mal_percentage)))
avg_belief_bc_mat = avg_belief_mat.copy()
pooling = True

for i, k in enumerate(pool_size):
    for j, malfunctioning in enumerate(mal_percentage):
        result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                        max_iteration=max_iteration, k=k, init_x=init_x, noise=noise,
                                        alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                        memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c,
                                        strategy=strategy,
                                        threshold=threshold, pooling=pooling)

        avg_belief_mat[i,j] = result['belief_avg_true_good'].mean(axis=1)[-1]

        result_bc = simulate_malicious_acc_bc(simulation_times=simulation_times, pop_n=pop_n,
                                        max_iteration=max_iteration, k=k, init_x=init_x, noise=noise,
                                        alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                        memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c,
                                        strategy=strategy,
                                        threshold=threshold, pooling=pooling)

        avg_belief_bc_mat[i,j] = result['belief_avg_true_good'].mean(axis=1)[-1]
avg_belief_diff = avg_belief_mat-avg_belief_bc_mat
plt.figure()
sns.heatmap(pd.DataFrame(avg_belief_diff, index=pool_size, columns=mal_percentage), vmin=-1, vmax=1)
plt.title("diff of avg belief (evidence rate=0.01, noise=0.2, threshold=0.5)")
plt.ylabel("pool size k")
plt.xlabel("percentage of malicious agents")
plt.savefig(file_name + 'heatmap_avg_belief_bc_k_malicious.png')

