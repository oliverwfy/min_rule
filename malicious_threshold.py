from model import *
import matplotlib.pyplot as plt
import warnings


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
prob_evidence = 0.02


# percentage of malicious agents
malicious = 0.2

threshold= 0.5

# deception
strategy = 'deception'

# dampening
dampening = 0.01


file_name = 'malicious_threshold/'


mal_c_ls = [None, 0.1, 0.3, 0.5, 0.7, 0.9]
threshold_ls = [0.1, 0.3, 0.5,0.7, 0.9]
pool_size = [3, 6, 9, 12, 15]


# mal_c, threshold
k = 6
for mal_c in mal_c_ls:
    for threshold in threshold_ls:

        result = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=True)

        belief_avg_true_good = result['belief_avg_true_good'].mean(axis=1)
        plt.figure(f'belief avg good c{mal_c}')
        plt.plot(range(max_iteration), belief_avg_true_good)

    if mal_c:
        plt.axhline(y=mal_c, color='gray', linestyle='--')
    plt.title(f'average belief against iteration (k={k}, {int(malicious*100)}% malicious agents, c={mal_c})')
    plt.xlabel('iteration')
    plt.ylabel('average belief')
    plt.ylim([0,1])
    legend = ['confidence threshold = ' + str(gamma) for gamma in threshold_ls] + [f'c = {mal_c}']
    plt.legend(legend)
    plt.savefig(file_name + f'threshold_mal_c_{mal_c}_k_{k}.png')



# mal_c, threshold
mal_c = 0.5
for k in pool_size:
    for threshold in threshold_ls:

        result = simulate_malicious(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=True)

        belief_avg_true_good = result['belief_avg_true_good'].mean(axis=1)
        plt.figure(f'belief avg good k {k}')
        plt.plot(range(max_iteration), belief_avg_true_good)
    if mal_c:
        plt.axhline(y=mal_c, color='gray', linestyle='--')
    plt.title(f'average belief against iteration (k={k}, {int(malicious*100)}% malicious agents, c={mal_c})')
    plt.xlabel('iteration')
    plt.ylabel('average belief')
    plt.ylim([0,1])
    legend = ['confidence threshold = ' + str(gamma) for gamma in threshold_ls] + [f'c = {mal_c}']
    plt.legend(legend)
    plt.savefig(file_name + f'threshold_mal_k_{k}_c_{mal_c}.png')

