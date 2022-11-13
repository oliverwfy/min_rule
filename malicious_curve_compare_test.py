from model import simulate_malicious
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import seaborn as sns


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
simulation_times = 1

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.02


# percentage of malicious agents
malicious = 0.1


file_name = 'malicious_compare_test/'

threshold= 0.5
pool_size = [2, 3, 6, 9, 12, 15]
malicious_ls = np.linspace(0, 0.2, 6)

# avg belief against iteration in different pool size
malicious = 0.2

mal_c = 0.3

stratefy = 'deception'

for k in pool_size:
    stratefy = 'deception'
    result_extreme = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                                          max_iteration=max_iteration, k=k, init_x = init_x,
                                          alpha=alpha, prob_evidence=prob_evidence, mal_c=None,
                                          memory=False, memory_weight=0.5, malicious=malicious, strategy=stratefy,
                                          threshold=threshold, pooling=True)
    plt.figure()
    belief_extreme = result_extreme['belief_avg_true_good'].mean(axis=1)
    plt.plot(range(max_iteration), belief_extreme)

    stratefy = 'deception'
    result_deception = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                      max_iteration=max_iteration, k=k, init_x = init_x,
                      alpha=alpha, prob_evidence=prob_evidence, mal_c=mal_c,
                      memory=False, memory_weight=0.5, malicious=malicious, strategy=stratefy,
                      threshold=threshold, pooling=True)
    belief_deception = result_deception['belief_avg_true_good'].mean(axis=1)
    plt.plot(range(max_iteration), belief_deception)

    stratefy = 'pull'
    result_pull = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                                max_iteration=max_iteration, k=k, init_x = init_x,
                                alpha=alpha, prob_evidence=prob_evidence,mal_c=mal_c,
                                memory=False, memory_weight=0.5, malicious=malicious, strategy=stratefy,
                                threshold=threshold, pooling=True)
    belief_pull = result_pull['belief_avg_true_good'].mean(axis=1)
    plt.plot(range(max_iteration), belief_pull)

    plt.title(f'average belief against iteration ({int(malicious*100)}% malicious agents)')
    plt.xlabel('iteration')
    plt.ylim([0,1])
    plt.ylabel('average belief')
    legend = ['extreme disturbance with deception', 'pull with deception', 'pull only']
    plt.legend(legend)
    plt.savefig(file_name + f'belief_avg_true_good_ag_iteration_compare_k_{k}.png')


