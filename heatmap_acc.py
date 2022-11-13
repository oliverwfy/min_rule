import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import simulate
import warnings

warnings.filterwarnings('ignore')
font_size = 16

# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = 0.3

# number of pooled agents in each iteration
k = 5

# maximum iteration
max_iteration = 2000

# simulation times
simulation_times = 100

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.02

# percentage of malfunctioning agents
malfunctioning = 0.10

# threshold for fault detection
threshold = 0.5


file_name = 'heatmap_acc/'


mal_percentage = np.arange(0,51,5) / 100.0
pool_size = [20, 15, 10, 6, 3, 2, 1]
mal_x_ls = np.round(np.linspace(0.05, 0.6, 12), 2)

# change : ["pool size", "mal_percentage"]
acc_mat = np.zeros((len(pool_size), len(mal_percentage)))
pre_mat = acc_mat.copy()
rec_mat = acc_mat.copy()

for i, k in enumerate(pool_size):
    for j, malfunctioning in enumerate(mal_percentage):


        result = simulate(simulation_times=simulation_times, pop_n=pop_n,
                          max_iteration=max_iteration, k=k, init_x = init_x,
                          mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                          memory=False, memory_weight=0.5, malfunctioning=malfunctioning,threshold=threshold,
                          pooling=True)

        acc_mat[i,j] = result['accuracy_avg'].mean(axis=1)[-1]
        pre_mat[i,j] = result['precision_avg'].mean(axis=1)[-1]
        rec_mat[i,j] = result['recall_avg'].mean(axis=1)[-1]

plt.figure()
sns.heatmap(pd.DataFrame(acc_mat, index=pool_size, columns=mal_percentage), vmin=1-mal_percentage.max(), vmax=1)
plt.title("accuracy (malfunctioning belief = 0.3)")
plt.ylabel("pool size k")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'acc_k_malfunctioning_iteration_{max_iteration}.png')

plt.figure()
sns.heatmap(pd.DataFrame(pre_mat, index=pool_size, columns=mal_percentage), vmin=0, vmax=1)
plt.title("precision (malfunctioning belief = 0.3)")
plt.ylabel("pool size k")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'precision_k_malfunctioning_iteration_{max_iteration}.png')

plt.figure()
sns.heatmap(pd.DataFrame(rec_mat, index=pool_size, columns=mal_percentage), vmin=0, vmax=1)
plt.title("recall (malfunctioning belief = 0.3)")
plt.ylabel("pool size k")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'recall_k_malfunctioning_iteration_{max_iteration}.png')




# change : ["pool size", "mal_x"]
acc_mat = np.zeros((len(pool_size), len(mal_x_ls)))
pre_mat = acc_mat.copy()
rec_mat = acc_mat.copy()

for i, k in enumerate(pool_size):
    for j, mal_x in enumerate(mal_x_ls):


        result = simulate(simulation_times=simulation_times, pop_n=pop_n,
                          max_iteration=max_iteration, k=k, init_x = init_x,
                          mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence, threshold=threshold,
                          memory=False, memory_weight=0.5, malfunctioning=malfunctioning,
                          pooling=True)

        acc_mat[i,j] = result['accuracy_avg'].mean(axis=1)[-1]
        pre_mat[i,j] = result['precision_avg'].mean(axis=1)[-1]
        rec_mat[i,j] = result['recall_avg'].mean(axis=1)[-1]

plt.figure()
sns.heatmap(pd.DataFrame(acc_mat, index=pool_size, columns=mal_x_ls), vmin=1-malfunctioning, vmax=1)
plt.title("accuracy (10% malfunctioning agents)")
plt.ylabel("pool size k")
plt.xlabel("malfunctioning initial belief")
plt.savefig(file_name + f'acc_k_mal_x_iteration_{max_iteration}.png')

plt.figure()
sns.heatmap(pd.DataFrame(pre_mat, index=pool_size, columns=mal_x_ls), vmin=0, vmax=1)
plt.title("precision (10% malfunctioning agents)")
plt.ylabel("pool size k")
plt.xlabel("malfunctioning initial belief")
plt.savefig(file_name + f'precision_k_mal_x_iteration_{max_iteration}.png')

plt.figure()
sns.heatmap(pd.DataFrame(rec_mat, index=pool_size, columns=mal_x_ls), vmin=0, vmax=1)
plt.title("recall (10% malfunctioning agents)")
plt.ylabel("pool size k")
plt.xlabel("malfunctioning initial belief")
plt.savefig(file_name + f'recall_k_mal_x_iteration_{max_iteration}.png')


# change : ["mal_x", "mal_percentage"]
acc_mat = np.zeros((len(mal_x_ls), len(mal_percentage)))
pre_mat = acc_mat.copy()
rec_mat = acc_mat.copy()


for i, mal_x in enumerate(mal_x_ls):
    for j, malfunctioning in enumerate(mal_percentage):
        result = simulate(simulation_times=simulation_times, pop_n=pop_n,
                          max_iteration=max_iteration, k=k, init_x = init_x,
                          mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
                          memory=False, memory_weight=0.5, malfunctioning=malfunctioning,threshold=threshold,
                          pooling=True)

        acc_mat[i,j] = result['accuracy_avg'].mean(axis=1)[-1]
        pre_mat[i,j] = result['precision_avg'].mean(axis=1)[-1]
        rec_mat[i,j] = result['recall_avg'].mean(axis=1)[-1]

plt.figure()
sns.heatmap(pd.DataFrame(acc_mat, index=mal_x_ls, columns=mal_percentage), vmin=1-mal_percentage.max(), vmax=1)
plt.title("accuracy (pool size k = 5)")
plt.ylabel("malfunctioning belief")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'acc_mal_x_mal_percent_iteration_{max_iteration}.png')

plt.figure()
sns.heatmap(pd.DataFrame(pre_mat, index=mal_x_ls, columns=mal_percentage), vmin=0, vmax=1)
plt.title("precision (pool size k = 5)")
plt.ylabel("malfunctioning belief")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'precision_mal_x_mal_percent_iteration_{max_iteration}.png')

plt.figure()
sns.heatmap(pd.DataFrame(rec_mat, index=mal_x_ls, columns=mal_percentage), vmin=0, vmax=1)
plt.title("recall (pool size k = 5)")
plt.ylabel("malfunctioning belief")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'recall_mal_x_mal_percent_iteration_{max_iteration}.png')


#
# # change : ["pool size", "mal_x"] with memory 0.5
# memory_weight = 0.5
# acc_mat = np.zeros((len(pool_size), len(mal_x_ls)))
# pre_mat = acc_mat.copy()
# rec_mat = acc_mat.copy()
#
# for i, k in enumerate(pool_size):
#     for j, mal_x in enumerate(mal_x_ls):
#
#
#         result = simulate(simulation_times=simulation_times, pop_n=pop_n,
#                           max_iteration=max_iteration, k=k, init_x = init_x,
#                           mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
#                           memory=True, memory_weight=memory_weight, malfunctioning=malfunctioning,
#                           pooling=True)
#
#         acc_mat[i,j] = result['accuracy_avg'].mean(axis=1)[-1]
#         pre_mat[i,j] = result['precision_avg'].mean(axis=1)[-1]
#         rec_mat[i,j] = result['recall_avg'].mean(axis=1)[-1]
#
# plt.figure()
# sns.heatmap(pd.DataFrame(acc_mat, index=pool_size, columns=mal_x_ls), vmin=1-malfunctioning, vmax=1)
# plt.title("accuracy (10% malfunctioning agents)")
# plt.ylabel("pool size k")
# plt.xlabel("malfunctioning initial belief")
# plt.savefig(file_name + f'acc_k_mal_x_iteration_{max_iteration}_memory_{memory_weight}.png')
#
# plt.figure()
# sns.heatmap(pd.DataFrame(pre_mat, index=pool_size, columns=mal_x_ls), vmin=0, vmax=1)
# plt.title("precision (10% malfunctioning agents)")
# plt.ylabel("pool size k")
# plt.xlabel("malfunctioning initial belief")
# plt.savefig(file_name + f'precision_k_mal_x_iteration_{max_iteration}_memory_{memory_weight}.png')
#
# plt.figure()
# sns.heatmap(pd.DataFrame(rec_mat, index=pool_size, columns=mal_x_ls), vmin=0, vmax=1)
# plt.title("recall (10% malfunctioning agents)")
# plt.ylabel("pool size k")
# plt.xlabel("malfunctioning initial belief")
# plt.savefig(file_name + f'recall_k_mal_x_iteration_{max_iteration}_memory_{memory_weight}.png')
#
#
#
#
# # change : ["pool size", "mal_x"] with memory 0.9
# memory_weight = 0.9
# acc_mat = np.zeros((len(pool_size), len(mal_x_ls)))
# pre_mat = acc_mat.copy()
# rec_mat = acc_mat.copy()
#
# for i, k in enumerate(pool_size):
#     for j, mal_x in enumerate(mal_x_ls):
#
#
#         result = simulate(simulation_times=simulation_times, pop_n=pop_n,
#                           max_iteration=max_iteration, k=k, init_x = init_x,
#                           mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence,
#                           memory=True, memory_weight=memory_weight, malfunctioning=malfunctioning,
#                           pooling=True)
#
#         acc_mat[i,j] = result['accuracy_avg'].mean(axis=1)[-1]
#         pre_mat[i,j] = result['precision_avg'].mean(axis=1)[-1]
#         rec_mat[i,j] = result['recall_avg'].mean(axis=1)[-1]
#
# plt.figure()
# sns.heatmap(pd.DataFrame(acc_mat, index=pool_size, columns=mal_x_ls), vmin=1-malfunctioning, vmax=1)
# plt.title("accuracy (10% malfunctioning agents)")
# plt.ylabel("pool size k")
# plt.xlabel("malfunctioning initial belief")
# plt.savefig(file_name + f'acc_k_mal_x_iteration_{max_iteration}_memory_{memory_weight}.png')
#
# plt.figure()
# sns.heatmap(pd.DataFrame(pre_mat, index=pool_size, columns=mal_x_ls), vmin=0, vmax=1)
# plt.title("precision (10% malfunctioning agents)")
# plt.ylabel("pool size k")
# plt.xlabel("malfunctioning initial belief")
# plt.savefig(file_name + f'precision_k_mal_x_iteration_{max_iteration}_memory_{memory_weight}.png')
#
# plt.figure()
# sns.heatmap(pd.DataFrame(rec_mat, index=pool_size, columns=mal_x_ls), vmin=0, vmax=1)
# plt.title("recall (10% malfunctioning agents)")
# plt.ylabel("pool size k")
# plt.xlabel("malfunctioning initial belief")
# plt.savefig(file_name + f'recall_k_mal_x_iteration_{max_iteration}_memory_{memory_weight}.png')
#
#
