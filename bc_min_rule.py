from model import simulate_malicious_acc, simulate_malicious_acc_bc
import matplotlib.pyplot as plt
import warnings

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
max_iteration = 1000

# simulation times
simulation_times = 100

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

mal_c = 0.5

noise = 0.2
# dampening
dampening = 0.001

strategy = 'deception'

malicious_ls = [0, 0.05, 0.1, 0.15, 0.2]

for malicious in malicious_ls:
    pooling = False


    result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)

    plt.figure(f'avg belief {malicious}')
    belief = result['belief_avg_true_good'].mean(axis=1)
    plt.plot(range(max_iteration), belief, '--')

    pooling = True


    result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)

    plt.figure(f'avg belief {malicious}')

    belief = result['belief_avg_true_good'].mean(axis=1)
    plt.plot(range(max_iteration), belief)

    result = simulate_malicious_acc_bc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,noise = noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)
    plt.figure(f'avg belief {malicious}')

    belief = result['belief_avg_true_good'].mean(axis=1)
    plt.plot(range(max_iteration), belief)


    plt.figure(f'avg belief {malicious}')

    plt.title(f'malicious = {malicious}, evidence rate = {prob_evidence}, noise = {noise}, threshold={threshold}')
    plt.xlabel('iteration')
    plt.ylabel('avg belief')
    legend = ['evidence only', 'confidence updating', 'BC']
    plt.legend(legend)

    plt.savefig(file_name+f'avg_belief_bc_model_malicious_{malicious}.png')


