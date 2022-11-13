from model import simulate_malicious_acc, simulate_malfunctioning
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
prob_evidence = 0.02
weights_updating = 'kl_divergence'


# percentage of malicious agents
malicious = 0.1


file_name = 'malicious_acc/'


threshold= 0.5
k = 10
malicious_ls = np.linspace(0, 0.2, 6)

# avg belief against iteration in different pool size
malicious = 0.1

mal_c = 0.5


# dampening
dampening = 0.001

strategy = 'deception'

mal_c_ls = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9]

#
# malicious_ls = [None, 0.1, 0.2, 0.3]
#
#
# for malicious in malicious_ls:
#     result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
#                                 max_iteration=max_iteration, k=k, init_x = init_x,
#                                 alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
#                                 memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
#                                 threshold=threshold, pooling=True)
#
#
#     acc = result['accuracy_avg'].mean(axis=1)
#
#     mal_x = 0.3
#
#     result_malfunction = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n, max_iteration=max_iteration, k=k, init_x = init_x, mal_x = mal_x, equal_weights=False,
#                  alpha=alpha, prob_evidence=prob_evidence, malfunctioning=malicious, threshold= threshold, weights_updating = weights_updating, dampening=None,
#                  noise=None, pooling=True, memory=False, memory_weight=0.5, consensus_only=False, detection_only=False)
#
#     acc_malfunctioning = result_malfunction['accuracy_avg'].mean(axis=1)
#
#     plt.figure(f'acc (k = {k} malicious_{malicious}')
#
#     plt.plot(range(max_iteration), acc_malfunctioning)
#     plt.plot(range(max_iteration), acc)
#
#     plt.title(f'accuracy against iteration ({int(malicious*100)}% malicious agents)')
#     plt.xlabel('iteration')
#     plt.ylabel('accuracy')
#     legend = [f'fixed_belief = {mal_x}', 'min_rule']
#     plt.ylim([0.5, 1])
#     plt.legend(legend)
#     plt.savefig(file_name+f'acc_ag_iteration_two_strategy_k_{k}_malicious_{malicious}.png')
#
#     plt.figure(f'avg belief malicious_{malicious}')
#
#     belief = result['belief_avg_true_good'].mean(axis=1)
#     belief_malfunctioning = result_malfunction['belief_avg_true_good'].mean(axis=1)
#
#     plt.plot(range(max_iteration), belief_malfunctioning)
#     plt.plot(range(max_iteration), belief)
#
#     plt.title(f'avg belief against iteration ({int(malicious*100)}% malicious agents)')
#     legend = [f'fixed_belief = {mal_x}', 'min_rule']
#     plt.legend(legend)
#     plt.ylim([0, 1])
#     plt.savefig(file_name+f'avg_belief_ag_iteration_two_strategy_k_{k}_malicious_{malicious}.png')
#
#
# plt.show()
#
#
#
#



noise = None

pool_size_ls = [3, 5, 10, 15, 20]
malicious_ls = [None, 0.1, 0.2, 0.3]

for k in pool_size_ls:

    pooling = True

    for malicious in malicious_ls:
        result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                        max_iteration=max_iteration, k=k, init_x = init_x, noise=noise,
                                        alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                        memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                        threshold=threshold, pooling=pooling)

        acc = result['accuracy_avg'].mean(axis=1)

        plt.figure(f'acc (k = {k}, noise={noise})')


        plt.plot(range(max_iteration), acc)

        plt.figure(f'avg belief malicious k={k}')
        belief = result['belief_avg_true_good'].mean(axis=1)
        plt.plot(range(max_iteration), belief)


    pooling = False

    result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
                                    max_iteration=max_iteration, k=k, init_x = init_x, noise=noise,
                                    alpha=alpha, prob_evidence=prob_evidence, dampening=dampening,
                                    memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
                                    threshold=threshold, pooling=pooling)


    acc = result['accuracy_avg'].mean(axis=1)
    plt.figure(f'acc (k = {k}, noise={noise})')
    plt.plot(range(max_iteration), acc)

    plt.figure(f'avg belief malicious k={k}')
    belief = result['belief_avg_true_good'].mean(axis=1)
    plt.plot(range(max_iteration), belief, '--')



    plt.figure(f'acc (k = {k}, noise={noise})')
    plt.title(f'accuracy against iteration (min_rule) k={k}')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    legend = ['0', '10%', '20%', '30%', 'evidence only']
    plt.legend(legend)
    plt.savefig(file_name+f'acc_ag_iteration_min_rule_k_{k}_noise_{noise}.png')




    plt.figure(f'avg belief malicious k={k}')
    plt.title(f'avg belief against iteration (min_rule) k={k}')
    plt.legend(legend)
    plt.ylim([0, 1])
    plt.savefig(file_name+f'avg_belief_ag_iteration_min_rule_k_{k}_noise_{noise}.png')


















# strategy = 'pull'
    #
    # for mal_c in mal_c_ls:
    #     for k in pool_size:
    #
    #         result = simulate_malicious_acc(simulation_times=simulation_times, pop_n=pop_n,
    #                                     max_iteration=max_iteration, k=k, init_x = init_x,
    #                                     alpha=alpha, prob_evidence=prob_evidence,dampening=dampening,
    #                                     memory=False, memory_weight=0.5, malicious=malicious, mal_c=mal_c, strategy=strategy,
    #                                     threshold=threshold, pooling=True)
    #
    #         acc = result['accuracy_avg'].mean(axis=1)
    #         plt.figure(f'acc c{mal_c} k')
    #         plt.plot(range(max_iteration), acc)
    #
    #     plt.title(f'accuracy against iteration ({int(malicious*100)}% malicious agents, c={mal_c})')
    #     plt.xlabel('iteration')
    #     plt.ylabel('accuracy')
    #     legend = ['k = ' + str(k) for k in pool_size]
    #     plt.legend(legend)
    #     plt.savefig(file_name + f'acc_ag_iteration_mal_c_{mal_c}_k_pull.png')
    #

