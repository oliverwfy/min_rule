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
max_iteration = 3000

# simulation times
simulation_times = 10

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.02

# percentage of malfunctioning agents
malfunctioning = 0.10

# threshold for fault detection
threshold = 0.5

# weights_updating
weights_updating = 'kl_divergence'


file_name = 'heatmap_detection/'
file_np = file_name+ 'npy/'

dampening = None

mal_percentage = np.arange(0,41,2) / 100.0
pool_size = np.arange(14)+2
mal_x_ls = np.round(np.linspace(0.1, 0.9, 17), 2)


# change : ["pool size", "mal_percentage"]
detection_mat = np.zeros((len(pool_size), len(mal_percentage)))

for i, k in enumerate(np.flip(pool_size)):
    for j, malfunctioning in enumerate(mal_percentage):


        result = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n,
                          max_iteration=max_iteration, k=k, init_x = init_x, weights_updating=weights_updating,dampening=dampening,
                          mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence, detection_only=True,
                          memory=False, memory_weight=0.5, malfunctioning=malfunctioning, threshold=threshold,
                          pooling=True)

        detection_mat[i,j] = result['detection_time'].mean()

np.save(file_np + f'detection_k_malfunctioning_iteration_{max_iteration}_{weights_updating}.npy', detection_mat)

plt.figure()
sns.heatmap(pd.DataFrame(detection_mat, index=np.flip(pool_size), columns=mal_percentage), vmin=0, vmax=max_iteration)
plt.title(f"time for detection (malfunctioning belief = {mal_x})")
plt.ylabel("pool size k")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'detection_k_malfunctioning_iteration_{max_iteration}_{weights_updating}.png')




# change : ["pool size", "mal_x"]
malfunctioning = 0.10
detection_mat = np.zeros((len(pool_size), len(mal_x_ls)))


for i, k in enumerate(np.flip(pool_size)):
    for j, mal_x in enumerate(mal_x_ls):


        result = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n,
                          max_iteration=max_iteration, k=k, init_x = init_x,weights_updating=weights_updating,dampening=dampening,
                          mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence, detection_only=True,
                          memory=False, memory_weight=0.5, malfunctioning=malfunctioning, threshold=threshold,
                          pooling=True)

        detection_mat[i,j] = result['detection_time'].mean()

np.save(file_np + f'detection_k_mal_x_iteration_{max_iteration}_{weights_updating}.npy', detection_mat)

plt.figure()
sns.heatmap(pd.DataFrame(detection_mat, index=np.flip(pool_size), columns=mal_x_ls), vmin=0, vmax=max_iteration)
plt.title("time for detection (10% malfunctioning agents)")
plt.ylabel("pool size k")
plt.xlabel("malfunctioning initial belief")
plt.savefig(file_name + f'detection_k_mal_x_iteration_{max_iteration}_{weights_updating}.png')



# change : ["mal_x", "mal_percentage"]
k = 5
detection_mat = np.zeros((len(mal_x_ls), len(mal_percentage)))
mal_x_ls = np.flip(mal_x_ls)

for i, mal_x in enumerate(mal_x_ls):
    for j, malfunctioning in enumerate(mal_percentage):
        result = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n,
                          max_iteration=max_iteration, k=k, init_x = init_x, weights_updating=weights_updating,dampening=dampening,
                          mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence, detection_only=True,
                          memory=False, memory_weight=0.5, malfunctioning=malfunctioning, threshold=threshold,
                          pooling=True)

        detection_mat[i,j] = result['detection_time'].mean()


np.save(file_np + f'detection_mal_x_mal_percent_iteration_{max_iteration}_{weights_updating}.npy', detection_mat)


plt.figure()
sns.heatmap(pd.DataFrame(detection_mat, index=mal_x_ls, columns=mal_percentage), vmin=0, vmax=max_iteration)
plt.title(f"time for detection (pool size k = {k})")
plt.ylabel("malfunctioning belief")
plt.xlabel("percentage of malfunctioning agents")
plt.savefig(file_name + f'detection_mal_x_mal_percent_iteration_{max_iteration}_{weights_updating}.png')

