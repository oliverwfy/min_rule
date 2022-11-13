import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import simulate_malicious
import warnings
from utility import *
from scipy.optimize import fsolve, zeros, newton, bisect, toms748
import math
from agent import *

warnings.filterwarnings('ignore')
font_size = 20

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
simulation_times = 2

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.02

# percentage of malfunctioning agents
malfunctioning = 0.20

# threshold for fault detection
threshold = 0.5

from time import time



def is_not_convex(f,num_variables,bounds=None,max_time=1,min_points=1000):
    """
    Input:
        f: a callable function, should take arguments of the form (n,x) where n is the number of samples
        and return a scalar
        num_variables: the number of variables in x
        bounds: optional (2,num_variables) array of bounds for the input variables
    Output:
        result: bool, true if the function is not convex
    Raises:
        TimeoutError
    """
    t0 = time()
    while time()-t0 < max_time:
        if bounds is None:
            pts = np.random.randn(2,min_points,num_variables)
        else:
            pts = (np.random.rand(2,min_points,num_variables)+bounds[0])*(bounds[1]/(1+bounds[0]))
        mean_pt = np.mean(pts,axis=0)
        if np.any((f(pts[0])+f(pts[1]))/2<f(mean_pt)):
            return True
    raise TimeoutError(f"Could not find any counterexamples, function \"{f.__name__}\" may be convex.")

def is_not_concave(f,num_variables,bounds=None,max_time=1,min_points=1000):
    """
    Input:
        f: a callable function, should take arguments of the form (n,x) where n is the number of samples
        and return a scalar
        num_variables: the number of variables in x
        bounds: optional (2,num_variables) array of bounds for the input variables
    Output:
        result: bool, true if the function is not concave
    Raises:
        TimeoutError
    """
    negf = lambda x : -f(x)
    try:
        return is_not_convex(negf,num_variables,bounds,max_time,min_points)
    except:
        raise TimeoutError(f"Could not find any counterexamples, function \"{f.__name__}\" may be concave.")

# x_ls = np.linspace(0.01,0.99, 100)
# pooled_prob = np.empty(x_ls.shape)
# confidence = np.empty(x_ls.shape)
#
# for i, x in enumerate(x_ls):
#     pool_prob = [0.8, 0.85, 0.9]
#     pool_prob.append(x)
#
#     weights = np.array([0.2, 0.3, 0.3, 0.2])
#
#     pooled_prob[i] = log_op(np.array(pool_prob), weights)
#     confidence[i] = confidence_updating(pool_prob, pooled_prob[i])[-1]
#
# plt.plot(x_ls, pooled_prob)
# plt.plot(x_ls, confidence, color='black')
# plt.xlabel('belief of malicious agent')
# plt.ylabel('pooled belief / confidence')
# plt.title('deception in opinion pooling of a1')
# plt.axhline(y=0.5, color='r', linestyle='-')
# plt.fill_between(x_ls, 0.5, confidence, color='green',where=(confidence > 0.5), alpha=0.2)
# plt.fill_between(x_ls, 0.5, confidence, color='red',where=(confidence < 0.5), alpha=0.2)
# plt.legend(['pooled probability in agent a1', 'confidence for malicious agent', 'confidence threshold = 0.5'])
# plt.savefig('fig/deception.png')
# plt.show()


# x_ls = np.linspace(0.01,0.99, 1000)
# pooled_prob = np.empty(x_ls.shape)
# confidence = np.empty(x_ls.shape)
#
# pool = [0.9, 0.9, 0.9]
# w = [0.2, 0.3, 0.1]
#
#
# for i, x in enumerate(x_ls):
#     pool_prob = pool.copy()
#     pool_prob.append(x)
#     pool_prob = np.array(pool_prob)
#     weights = w.copy()
#     weights.append(1-sum(weights))
#     weights = np.array(weights)
#     pooled_prob[i] = log_op(pool_prob, weights)
#     confidence[i] = confidence_updating(pool_prob, pooled_prob[i])[-1]
#
# plt.plot(x_ls, confidence-threshold, color='black')
# plt.axhline(y=0, color='r', linestyle='-')
# plt.xlabel('belief of malicious agent')
# plt.ylabel('pooled belief / confidence')
# plt.title('deception in opinion pooling of a1')
#
#
# plt.show()
#
#
# def log_linear_deception(x, pool, w):
#     numerator = np.prod(pool ** w) * (x ** (1-w.sum()))
#     return numerator / (numerator + np.prod((1 - pool) ** w) * ((1 - x) ** (1-w.sum()) ))
#
# def func(x):
#
#     return np.exp(-kl_divergence(np.array(log_linear_deception(x, np.array(pool), np.array(w))), x)) - threshold
#
# root = newton(func, [0.02, 1])
# print(root)
# x = min(root)
# if not np.isnan(x):
#     x = math.ceil(x * 10e5) / 10e5
# else:
#     x = 0.001
# print(x)


# x_ls = np.random.uniform(0,1,(100,))
# u = np.zeros(x_ls.shape)
#
# a = 0.8
# b = 0.2
# for i, x in enumerate(x_ls):
#     if i == x_ls.size-1:
#         continue
#
#     u[i+1] = a*u[i] + b*(x_ls[i+1]-x) if (x_ls[i+1]-x) > 0 else b*u[i] + a*(x_ls[i+1]-x)
#
# plt.plot(np.arange(u.size), u)
# plt.show()

# malicious = 0.1
# pop = np.array([Agent(100, _, 0.5,state=True) for _ in range(100)])
# # malicious_id = np.arange(0, int(len(pop)*malicious))
# # for i in malicious_id:
# #     pop[i] = Malicious(pop[i].id, threshold)
#
# malicious_id = generate_malicious_agents(pop, malicious, threshold)
#


#
# pool_size = np.arange(14)+2
# malicious_ls = np.linspace(0, 0.3, 16)
# prob_evidence_ls = np.linspace(0, 0.1, 21)
#
# file_name = 'malicious_heatmap/npy/'
# np_1 = np.load(file_name+'belief_avg_k_evidence_rate_pull_mal_c_0.2.npy')
# np_2 = np.load(file_name+'belief_avg_k_evidence_rate_deception_mal_c_0.2.npy')
# np_3 = np.load(file_name+'belief_avg_k_evidence_rate_pull_mal_c_0.5.npy')
# np_4 = np.load(file_name+'belief_avg_k_evidence_rate_deception_mal_c_0.5.npy')
# np_5 = np.load(file_name+'belief_avg_k_evidence_rate_pull_mal_c_0.8.npy')
# np_6 = np.load(file_name+'belief_avg_k_evidence_rate_deception_mal_c_0.8.npy')
#
#
#
# f,axs = plt.subplots(3,3, gridspec_kw={'width_ratios':[1,1,0.04]}, figsize=(12, 12))
#
#
# g1 = sns.heatmap(pd.DataFrame(np_1,index= np.flip(pool_size),columns = prob_evidence_ls),cmap='rocket', cbar=False,ax=axs[0,0],vmin=0, vmax=1, xticklabels=False)
# g1.set_ylabel('k', fontsize = font_size)
# g1.set_title(r'$ c=0.2, \rho=20\% $ (Raider)', fontsize=font_size)
#
# g2 = sns.heatmap(pd.DataFrame(np_2,index= np.flip(pool_size),columns = prob_evidence_ls),cmap='rocket', ax=axs[0,1], cbar_ax=axs[0,2], vmin=0, vmax=1, yticklabels=False, xticklabels=False)
# g2.set_title(r'$ c=0.2, \rho=20\% $ (Disguiser)', fontsize=font_size)
#
# g3 = sns.heatmap(pd.DataFrame(np_3,index= np.flip(pool_size),columns = prob_evidence_ls),cmap='rocket', cbar=False,ax=axs[1,0],vmin=0, vmax=1,xticklabels=False)
# g3.set_ylabel('k', fontsize = font_size)
# g3.set_title(r'$ c=0.5, \rho=20\% $ (Raider)', fontsize=font_size)
#
# g4 = sns.heatmap(pd.DataFrame(np_4,index= np.flip(pool_size),columns = prob_evidence_ls),cmap='rocket', ax=axs[1,1], cbar_ax=axs[1,2], vmin=0, vmax=1, yticklabels=False, xticklabels=False)
# g4.set_title(r'$ c=0.5, \rho=20\% $ (Disguiser)', fontsize=font_size)
#
# g5 = sns.heatmap(pd.DataFrame(np_5,index= np.flip(pool_size),columns = prob_evidence_ls),cmap='rocket', cbar=False,ax=axs[2,0],vmin=0, vmax=1,)
# g5.set_ylabel('k', fontsize = font_size)
# g5.set_xlabel(r'$ \beta $', fontsize = font_size)
# g5.set_title(r'$ c=0.8, \rho=20\% $ (Raider)', fontsize=font_size)
# g6 = sns.heatmap(pd.DataFrame(np_6,index= np.flip(pool_size),columns = prob_evidence_ls),cmap='rocket', ax=axs[2,1], cbar_ax=axs[2,2], vmin=0, vmax=1,yticklabels=False)
# g6.set_title(r' $c=0.8, \rho=20\% $ (Disguiser)', fontsize=font_size)
# g6.set_xlabel(r'$ \beta $', fontsize = font_size)
#
# plt.tight_layout()
# plt.savefig("test/malicious_heatmap_k_beta.png")
#
#
# file_name = 'malicious_heatmap/npy/'
# np_1 = np.load(file_name+'belief_avg_k_malicious_pull_mal_c_0.2.npy')
# np_2 = np.load(file_name+'belief_avg_k_malicious_deception_mal_c_0.2.npy')
# np_3 = np.load(file_name+'belief_avg_k_malicious_pull_mal_c_0.5.npy')
# np_4 = np.load(file_name+'belief_avg_k_malicious_deception_mal_c_0.5.npy')
# np_5 = np.load(file_name+'belief_avg_k_malicious_pull_mal_c_0.8.npy')
# np_6 = np.load(file_name+'belief_avg_k_malicious_deception_mal_c_0.8.npy')
#
#
# f,axs = plt.subplots(3,3, gridspec_kw={'width_ratios':[1,1,0.04]}, figsize=(12, 15))
#
#
# g1 = sns.heatmap(pd.DataFrame(np_1,index= np.flip(pool_size),columns = malicious_ls),cmap='rocket', cbar=False,ax=axs[0,0],vmin=0, vmax=1, xticklabels=False)
# g1.set_ylabel('k', fontsize = font_size)
# g1.set_title(r'$ c=0.2, \beta=0.02 $ (Raider)', fontsize=font_size)
#
# g2 = sns.heatmap(pd.DataFrame(np_2,index= np.flip(pool_size),columns = malicious_ls),cmap='rocket', ax=axs[0,1], cbar_ax=axs[0,2], vmin=0, vmax=1, yticklabels=False, xticklabels=False)
# g2.set_title(r'$ c=0.2, \beta=0.02 $ (Disguiser)', fontsize=font_size)
#
# g3 = sns.heatmap(pd.DataFrame(np_3,index= np.flip(pool_size),columns = malicious_ls),cmap='rocket', cbar=False,ax=axs[1,0],vmin=0, vmax=1,xticklabels=False)
# g3.set_ylabel('k', fontsize = font_size)
# g3.set_title(r'$ c=0.5, \beta=0.02 $ (Raider)', fontsize=font_size)
#
# g4 = sns.heatmap(pd.DataFrame(np_4,index= np.flip(pool_size),columns = malicious_ls),cmap='rocket', ax=axs[1,1], cbar_ax=axs[1,2], vmin=0, vmax=1, yticklabels=False, xticklabels=False)
# g4.set_title(r'$ c=0.5, \beta=0.02 $ (Disguiser)', fontsize=font_size)
#
# g5 = sns.heatmap(pd.DataFrame(np_5,index= np.flip(pool_size),columns = malicious_ls),cmap='rocket', cbar=False,ax=axs[2,0],vmin=0, vmax=1,)
# g5.set_ylabel('k', fontsize = font_size)
# g5.set_xlabel(r'$ \rho $', fontsize = font_size)
# g5.set_title(r'$ c=0.8, \beta=0.02 $ (Raider)', fontsize=font_size)
# g6 = sns.heatmap(pd.DataFrame(np_6,index= np.flip(pool_size),columns = malicious_ls),cmap='rocket', ax=axs[2,1], cbar_ax=axs[2,2], vmin=0, vmax=1,yticklabels=False)
# g6.set_title(r' $c=0.8, \beta=0.02 $ (Disguiser)', fontsize=font_size)
# g6.set_xlabel(r'$ \rho $', fontsize = font_size)
#
# plt.tight_layout()
# plt.savefig("test/malicious_heatmap_k_rho.png")
#
#
#
#
#
# np_1 = np.load(file_name+'belief_avg_evidence_rate_malicious_pull_mal_c_0.2.npy')
# np_2 = np.load(file_name+'belief_avg_evidence_rate_malicious_deception_mal_c_0.2.npy')
# np_3 = np.load(file_name+'belief_avg_evidence_rate_malicious_pull_mal_c_0.5.npy')
# np_4 = np.load(file_name+'belief_avg_evidence_rate_malicious_deception_mal_c_0.5.npy')
# np_5 = np.load(file_name+'belief_avg_evidence_rate_malicious_pull_mal_c_0.8.npy')
# np_6 = np.load(file_name+'belief_avg_evidence_rate_malicious_deception_mal_c_0.8.npy')
#
#
# f,axs = plt.subplots(3,3, gridspec_kw={'width_ratios':[1,1,0.04]}, figsize=(12, 22))
#
#
# g1 = sns.heatmap(pd.DataFrame(np_1,index= np.flip(prob_evidence_ls),columns = malicious_ls), cbar=False,ax=axs[0,0],cmap='rocket', vmin=0, vmax=1, xticklabels=False)
# g1.set_ylabel(r'$ \beta $', fontsize = font_size)
# g1.set_title(r'$ c=0.2, k=5 $ (Raider)', fontsize=font_size)
#
# g2 = sns.heatmap(pd.DataFrame(np_2,index= np.flip(prob_evidence_ls),columns = malicious_ls), ax=axs[0,1],cmap='rocket', cbar_ax=axs[0,2], vmin=0, vmax=1, yticklabels=False, xticklabels=False)
# g2.set_title(r'$ c=0.2, k=5 $ (Disguiser)', fontsize=font_size)
#
# g3 = sns.heatmap(pd.DataFrame(np_3,index= np.flip(prob_evidence_ls),columns = malicious_ls), cbar=False,cmap='rocket',ax=axs[1,0],vmin=0, vmax=1,xticklabels=False)
# g3.set_ylabel(r'$ \beta $', fontsize = font_size)
# g3.set_title(r'$ c=0.5, k=5 $ (Raider)', fontsize=font_size)
#
# g4 = sns.heatmap(pd.DataFrame(np_4,index= np.flip(prob_evidence_ls),columns = malicious_ls), ax=axs[1,1],cmap='rocket', cbar_ax=axs[1,2], vmin=0, vmax=1, yticklabels=False, xticklabels=False)
# g4.set_title(r'$ c=0.5, k=5 $ (Disguiser)', fontsize=font_size)
#
# g5 = sns.heatmap(pd.DataFrame(np_5,index= np.flip(prob_evidence_ls),columns = malicious_ls), cbar=False,cmap='rocket',ax=axs[2,0],vmin=0, vmax=1,)
# g5.set_ylabel(r'$ \beta $', fontsize = font_size)
# g5.set_xlabel(r'$ \rho $', fontsize = font_size)
# g5.set_title(r'$ c=0.8, k=5 $ (Raider)', fontsize=font_size)
# g6 = sns.heatmap(pd.DataFrame(np_6,index= np.flip(prob_evidence_ls),columns = malicious_ls), ax=axs[2,1],cmap='rocket', cbar_ax=axs[2,2], vmin=0, vmax=1,yticklabels=False)
# g6.set_title(r' $c=0.8, k=5 $ (Disguiser)', fontsize=font_size)
# g6.set_xlabel(r'$ \rho $', fontsize = font_size)
#
# plt.tight_layout()
#
# plt.savefig("test/malicious_heatmap_beta_rho.png")

#
#
#

fig_size = (18, 6)

mal_percentage = np.arange(0,41,2) / 100.0
pool_size = np.arange(14)+2
mal_x_ls = np.round(np.linspace(0.1, 0.9, 17), 2)


file_name = 'heatmap_detection/npy/'
np_1 = np.load(file_name+'detection_k_malfunctioning_iteration_3000_kl_divergence.npy')
np_2 = np.load(file_name+'detection_k_malfunctioning_iteration_3000_normal.npy')


f,axs = plt.subplots(1,3, gridspec_kw={'width_ratios':[1,1,0.08]}, figsize=(20,6))


g1 = sns.heatmap(pd.DataFrame(np_1,index= np.flip(pool_size),columns = mal_percentage), cbar=False,ax=axs[0],vmin=0, vmax=3000)
g1.set_xlabel(r"$ \rho $", fontsize = font_size)
g1.set_ylabel('k ', fontsize = font_size)
g1.set_title(r'$  x_m=0.4 $  (KL-divergence)', fontsize=font_size)

g2 = sns.heatmap(pd.DataFrame(np_2,index= np.flip(pool_size),columns = mal_percentage), ax=axs[1], cbar_ax=axs[2], vmin=0, vmax=3000, yticklabels=False)
g2.set_xlabel(r"$ \rho $", fontsize = font_size)
g2.set_title(r'$  x_m=0.4 $  (normal distribution)', fontsize=font_size)

plt.savefig("test/detection_heatmap_k_rho.png")




file_name = 'heatmap_detection/npy/'
np_1 = np.load(file_name+'detection_k_mal_x_iteration_3000_kl_divergence.npy')
np_2 = np.load(file_name+'detection_k_mal_x_iteration_3000_normal.npy')


f,axs = plt.subplots(1,3, gridspec_kw={'width_ratios':[1,1,0.08]}, figsize=fig_size)


g1 = sns.heatmap(pd.DataFrame(np_1,index= np.flip(pool_size),columns = mal_x_ls), cbar=False,ax=axs[0],vmin=0, vmax=3000)
g1.set_xlabel(r"$ x_m $", fontsize = font_size)
g1.set_ylabel('k ', fontsize = font_size)
g1.set_title(r'$  \rho=10\% $  (KL-divergence)', fontsize=font_size)

g2 = sns.heatmap(pd.DataFrame(np_2,index= np.flip(pool_size),columns = mal_x_ls), ax=axs[1], cbar_ax=axs[2], vmin=0, vmax=3000, yticklabels=False)
g2.set_xlabel(r"$ x_m $", fontsize = font_size)
g2.set_title(r'$  \rho=10\% $  (normal distribution)', fontsize=font_size)
plt.savefig("test/detection_heatmap_k_mal_x.png")



file_name = 'heatmap_detection/npy/'
np_1 = np.load(file_name+'detection_mal_x_mal_percent_iteration_3000_kl_divergence.npy')
np_2 = np.load(file_name+'detection_mal_x_mal_percent_iteration_3000_normal.npy')


f,axs = plt.subplots(1,3, gridspec_kw={'width_ratios':[1,1,0.08]}, figsize=fig_size)


g1 = sns.heatmap(pd.DataFrame(np_1,index= np.flip(mal_x_ls),columns = mal_percentage), cbar=False,ax=axs[0],vmin=0, vmax=3000)
g1.set_xlabel(r"$ \rho $", fontsize = font_size)
g1.set_ylabel(r'$x_m$', fontsize = font_size)
g1.set_title(r'$  k=5 $  (KL-divergence)', fontsize=font_size)

g2 = sns.heatmap(pd.DataFrame(np_2,index= np.flip(mal_x_ls),columns = mal_percentage), ax=axs[1], cbar_ax=axs[2], vmin=0, vmax=3000, yticklabels=False)
g2.set_xlabel(r"$ \rho $", fontsize = font_size)
g2.set_title(r'$  k=5 $  (normal distribution)', fontsize=font_size)

plt.savefig("test/detection_heatmap_x_m_rho.png")

