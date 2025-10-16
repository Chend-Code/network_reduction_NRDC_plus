"""
@File    :   utils.py
@Time    :   2025/4/28 14:31:10
@Author  :   chend
@Contact :   chend_zqfpu@163.com
"""
import networkx as nx
import numpy as np
import EoN
import random
from scipy.interpolate import interp1d
from scipy.integrate import simps
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use("science")
plt.rcParams.update({'font.size': 10})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.rc('font', family='Times New Roman')


# 节点重新编号：0~(N-1)
def relabel_nodes_labels(G):
    mapping = dict(zip(G, range(len(G.nodes()))))
    G = nx.relabel_nodes(G, mapping)
    return G


def cal_DC_plus(G):
    # 初始化存储每个节点 DC^+_i 的字典
    dc_plus_dict = {}

    # 遍历网络中的每个节点
    for node in G.nodes():
        # 获取节点的邻居集合
        neighbors = list(G.neighbors(node))
        # 计算节点的度 k_i
        k_i = G.degree(node)
        # 计算平均邻居度 k_nn,i
        if len(neighbors) == 0:
            k_nn_i = 0
        else:
            neighbor_degrees = [G.degree(neigh) for neigh in neighbors]
            k_nn_i = np.sum(neighbor_degrees) / len(neighbors)
        # 计算 DC^+_i
        dc_plus_i = k_i * k_nn_i
        dc_plus_dict[node] = dc_plus_i

    return dc_plus_dict


# 从初始网络中按sequence的值从小到大删除比例为q的节点
def remove_nodes(G, sequence, N, q):
    rn = int(N*q)
    G.remove_nodes_from(list(sequence.keys())[:rn])

    return G


def get_LCC(G):
    # 最大连通子图
    if not nx.is_connected(G):
        Gcc0 = sorted(nx.connected_components(G), key=len, reverse=True)
        # 得到图G的最大连通子图
        largest_cc0 = G.subgraph(Gcc0[0])
        G = largest_cc0.copy()
    # print(nx.is_connected(G))

    return G


# 定义获取节点坐标的函数
def get_node_coordinates(G):
    pos = {}
    x = nx.get_node_attributes(G, 'x')
    y = nx.get_node_attributes(G, 'y')
    for i in G.nodes():
        ix = x[i]
        iy = y[i]
        pos[i] = (ix, iy)  # 节点i的坐标

    return pos

def get_node_size(G):
    node_size = {}
    for i in G.nodes():
        node_size[i] = nx.get_node_attributes(G, 'size')[i]*0.6

    return node_size


# 网络的度分布熵S
def get_pdf_entropy(G):
    deg = dict(G.degree())
    all_k = sorted(list(set(deg.values())))  # 获取所有可能的度值
    data = np.array(list(deg.values()))
    avks = np.mean(data)
    N = len(G)
    nums = len(all_k)
    x = np.zeros(nums)
    Pk = np.zeros(nums)
    for index, ki in enumerate(all_k):
        c = 0
        for i in deg.keys():
            if deg[i] == ki:
                c += 1
        if ki>0:
            x[index] = ki/avks
            Pk[index] = c/N     
    pdf_entropy = np.sum(-Pk * np.log2(Pk))

    return pdf_entropy


# 互补累积度分布
def get_ccdf(G):
    deg = dict(G.degree()) 
    all_k = sorted(list(set(deg.values())))
    data = np.array(list(deg.values()))
    avks = np.mean(data)
    N = len(G)
    nums = len(all_k)
    x = np.zeros(nums)
    Pk = np.zeros(nums)
    for index, ki in enumerate(all_k):
        c = 0
        for i in deg.keys():
            if deg[i] == ki:
                c += 1
        if ki>0:
            x[index] = ki/avks
            Pk[index] = c/N

    Pck = np.array([sum(Pk[i:]) for i in range(len(Pk))])
    x[x == 0] = 'nan'
    Pck[Pck == 0] = 'nan'

    return x, Pck



def cal_avk_S_LCC(G0, sequence, N, q, avk0):
    y_avk = [1.0]
    y_S_LCC = [1.0]
    for i in range(len(q)):
        if i > 0:  # i=0时，q=0对应初始网络
            G = G0.copy()
            Gs = remove_nodes(G, sequence, N, q[i])
            avks = 2*nx.number_of_edges(Gs)/nx.number_of_nodes(Gs)
            y_avk.append(avks/avk0)
            
            Ns = nx.number_of_nodes(Gs)
            Gs_LCC = get_LCC(Gs)
            S_LCC = nx.number_of_nodes(Gs_LCC)/Ns
            y_S_LCC.append(S_LCC)        
            
    return np.array(y_avk), np.array(y_S_LCC)


def load_graph_data(filename):
    G = nx.read_edgelist("../datasets/" + filename + ".dat")
    G = relabel_nodes_labels(G)
    return G


def local_degree_sparsification(Gs, avk0, kmin):
    G = Gs.copy()
    flag = False
    while True:
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) <= kmin:     # 设定允许剪枝的节点度值的下限
                continue

            neighbors.sort(key=lambda x: G.degree(x), reverse=False) # 升序排序
            select_neighbor = neighbors[0]
            e = (node, select_neighbor)
            if e in G.edges():
                G.remove_edge(*e)

            # 检查移除该边后，图是否仍然连通
            if not nx.is_connected(G):
                # 如果不连通，恢复该边
                G.add_edge(*e)

            if np.abs(2*nx.number_of_edges(G)/nx.number_of_nodes(G)-avk0)<0.01:
                flag = True
                break
        if flag:
            break

    return G


# def cal_SIS_it(Gs, beta=1.0, gamma=1.0, tmin=0.0, tmax=4.0, iterations=100):
#     Ns = len(Gs.nodes())
#     ratio_N = int(0.1*Ns)
#     report_times = np.linspace(tmin, tmax, 51)
#     DC = dict(Gs.degree())
#     sorted_DC = dict(sorted(DC.items(), key=lambda x: x[1], reverse=True))
#     infected_nodes = list(sorted_DC.keys())[:ratio_N]

#     obs_I = 0 * report_times
#     for _ in range(iterations):
#         t, S, I = EoN.fast_SIS(Gs, beta, gamma, initial_infecteds=infected_nodes, tmax=tmax)
#         obs_I += EoN.subsample(report_times, t, I)
    
#     return obs_I * 1. / iterations / Ns


# def cal_SIS_it_vs_q(G0, sequence, N, q, kmin=None, prune_flag=False):
#     Fi = []
#     avk0 = 2*nx.number_of_edges(G0)/nx.number_of_nodes(G0)
#     for i in range(len(q)):
#         G = G0.copy()
#         Gs = remove_nodes(G, sequence, N, q[i])
#         avks = 2*nx.number_of_edges(Gs)/nx.number_of_nodes(Gs)
#         print("subnetwork:", avks, nx.number_of_nodes(Gs))

#         Gs = get_LCC(Gs)
#         if avks > avk0 and prune_flag:
#             Gs = local_degree_sparsification(Gs, avk0, kmin)
#             print(nx.is_connected(Gs))
#             avks = 2*nx.number_of_edges(Gs)/nx.number_of_nodes(Gs)
#             print("pruned subnetwork:", avks, nx.number_of_nodes(Gs))
#         Fii = cal_SIS_it(Gs)
#         Fi.append(Fii)
#     return Fi


def cal_SIR_rt_it(Gs, beta=1.0, gamma=1.0, tmin=0.0, tmax=6.0, iterations=100):
    Ns = len(Gs.nodes())
    ratio_N = int(0.1*Ns)
    report_times = np.linspace(tmin, tmax, 51)
    DC = dict(Gs.degree())
    sorted_DC = dict(sorted(DC.items(), key=lambda x: x[1], reverse=True))
    infected_nodes = list(sorted_DC.keys())[:ratio_N] # case1: 从Gs中选择度最大的前ratio_N个节点作为感染节点

    obs_R = 0 * report_times
    obs_I = 0 * report_times
    for _ in range(iterations):
        t, S, I, R = EoN.fast_SIR(Gs, beta, gamma, initial_infecteds=infected_nodes, tmax=tmax)
        obs_R += EoN.subsample(report_times, t, R)
        obs_I += EoN.subsample(report_times, t, I)
    
    return obs_R * 1. / iterations / Ns, obs_I * 1. / iterations / Ns


def cal_SIR_rt_it_vs_q(G0, sequence, N, q, kmin=None, prune_flag=False):
    Fr = []
    Fi = []
    avk0 = 2*nx.number_of_edges(G0)/nx.number_of_nodes(G0)
    for i in range(len(q)):
        G = G0.copy()
        Gs = remove_nodes(G, sequence, N, q[i])
        avks = 2*nx.number_of_edges(Gs)/nx.number_of_nodes(Gs)
        print("subnetwork:", avks, nx.number_of_nodes(Gs))

        Gs = get_LCC(Gs)
        if avks > avk0 and prune_flag:
            Gs = local_degree_sparsification(Gs, avk0, kmin)
            print(nx.is_connected(Gs))
            avks = 2*nx.number_of_edges(Gs)/nx.number_of_nodes(Gs)
            print("pruned subnetwork:", avks, nx.number_of_nodes(Gs))
        Fri, Fii = cal_SIR_rt_it(Gs)
        Fr.append(Fri)
        Fi.append(Fii)
    return Fr, Fi


# def cal_SIR_rt_it_case2(Gs, beta=1.0, gamma=1.0, tmin=0.0, tmax=6.0, iterations=100):
#     Ns = len(Gs.nodes())
#     ratio_N = int(0.1*Ns)
#     report_times = np.linspace(tmin, tmax, 51)

#     rt = 0 * report_times
#     it = 0 * report_times

#     samples = 10
#     for _ in range(samples):
#         infected_nodes = list(random.sample(list(Gs.nodes()), ratio_N)) # case2: 从Gs中随机选择ratio_N个节点作为感染节点
#         obs_R = 0 * report_times
#         obs_I = 0 * report_times
#         for _ in range(iterations):
#             t, S, I, R = EoN.fast_SIR(Gs, beta, gamma, initial_infecteds=infected_nodes, tmax=tmax)
#             obs_R += EoN.subsample(report_times, t, R)
#             obs_I += EoN.subsample(report_times, t, I)
#         rt += obs_R * 1. / iterations / Ns
#         it += obs_I * 1. / iterations / Ns
    
#     return rt/samples, it/samples


# def cal_SIR_rt_it_vs_q_case2(G0, sequence, N, q, kmin=None, prune_flag=False):
#     Fr = []
#     Fi = []
#     avk0 = 2*nx.number_of_edges(G0)/nx.number_of_nodes(G0)
#     for i in range(len(q)):
#         G = G0.copy()
#         Gs = remove_nodes(G, sequence, N, q[i])
#         avks = 2*nx.number_of_edges(Gs)/nx.number_of_nodes(Gs)
#         print("subnetwork:", avks, nx.number_of_nodes(Gs))

#         Gs = get_LCC(Gs)
#         if avks > avk0 and prune_flag:
#             Gs = local_degree_sparsification(Gs, avk0, kmin)
#             print(nx.is_connected(Gs))
#             avks = 2*nx.number_of_edges(Gs)/nx.number_of_nodes(Gs)
#             print("pruned subnetwork:", avks, nx.number_of_nodes(Gs))
#         Fri, Fii = cal_SIR_rt_it_case2(Gs)
#         Fr.append(Fri)
#         Fi.append(Fii)
#     return Fr, Fi



def cal_rho_r_vs_beta(Gs, beta_list, gamma=1.0, tmin=0.0, tmax=10.0, iterations=100):
    Ns = len(Gs.nodes())
    ratio_N = int(0.1*Ns)
    report_times = np.linspace(tmin, tmax, 51)
    DC = dict(Gs.degree())
    sorted_DC = dict(sorted(DC.items(), key=lambda x: x[1], reverse=True))
    infected_nodes = list(sorted_DC.keys())[:ratio_N]

    rho_R = []
    for beta in beta_list:
        R_nums = 0 * report_times
        for _ in range(iterations):
            t, S, I, R = EoN.fast_SIR(Gs, beta, gamma, initial_infecteds=infected_nodes, tmax=tmax)
            R_nums += EoN.subsample(report_times, t, R)
        rho_R_list = R_nums / iterations / Ns
        rho_R.append(rho_R_list[-1])

    return rho_R



def cal_rho_r_vs_beta_q(G0, sequence, N, q, beta_list, kmin=None, prune_flag=False):
    Fr = []
    avk0 = 2*nx.number_of_edges(G0)/nx.number_of_nodes(G0)
    for i in range(len(q)):
        G = G0.copy()
        Gs = remove_nodes(G, sequence, N, q[i])
        Gs = get_LCC(Gs)
        # print(nx.is_connected(Gs))
        avks = 2*len(Gs.edges())/len(Gs)
        if avks > avk0 and prune_flag:
            Gs = local_degree_sparsification(Gs, avk0, kmin)
        rho_Ri = cal_rho_r_vs_beta(Gs, beta_list)
        Fr.append(rho_Ri)
    return Fr


# 定义一个计算两条曲线重叠程度的函数
def compute_overlap(x, y1, y2):
    # 使用插值函数，使得两条曲线在相同的x点上进行比较
    f1 = interp1d(x, y1, kind='linear', fill_value='extrapolate')
    f2 = interp1d(x, y2, kind='linear', fill_value='extrapolate')
    
    # 选择更细的x点进行计算
    x_new = np.linspace(np.min(x), np.max(x), 500)
    
    # 获取插值后的y值
    y1_new = f1(x_new)
    y2_new = f2(x_new)
    
    # 计算两个曲线的差值
    diff = np.abs(y1_new - y2_new)
    
    # 使用Simpson公式计算曲线差异的积分
    area_diff = simps(diff, x_new)
    
    # 计算重叠程度，值越接近1表明重叠程度越好
    overlap = 1 / (1 + area_diff)
    return overlap



def cal_Z_tau(L, beta_range):
    ns = np.shape(L)[0]
    Z_tau = np.zeros(len(beta_range))
    # lambd, _ = np.linalg.eigh(L)
    lambd = np.linalg.eigvalsh(L)
    # print(lambd)
    for i, b in enumerate(beta_range):
        lrho = np.exp(-b * lambd)
        Z = lrho.sum()
        Z_tau[i] = Z

    return Z_tau



def cal_Z_tau_synthetic(G0, sequence, N, q, beta_range):
    nom_Z_tau = []
    for i in range(len(q)):
        G = G0.copy()
        Gs = remove_nodes(G, sequence, N, q[i])
        Gs = get_LCC(Gs)
        ns = nx.number_of_nodes(Gs)
        L = nx.laplacian_matrix(Gs).toarray()
        Z_tau_i = cal_Z_tau(L, beta_range)
        nom_Z_tau.append(Z_tau_i/ns)

    return nom_Z_tau



def cal_Z_tau_real(G0, sequence, N, q, beta_range, kmin=None, prune_flag=False):
    N0, M0 = nx.number_of_nodes(G0), nx.number_of_edges(G0)
    avk0 = 2*M0/N0
    nom_Z_tau = []
    for i in range(len(q)):
        G = G0.copy()
        Gs = remove_nodes(G, sequence, N, q[i])
        Gs = get_LCC(Gs)
        print(nx.is_connected(Gs))
        ns = nx.number_of_nodes(Gs)
        es = nx.number_of_edges(Gs)
        print(ns)
        avks = 2*es/ns
        print("1---:", avks)

        if avks > avk0 and prune_flag:
            Gs = local_degree_sparsification(Gs, avk0, kmin)
            ns = nx.number_of_nodes(Gs)
            es = nx.number_of_edges(Gs)
            print(ns)
            avks = 2*es/ns
            print("2---:", avks)

        L = nx.laplacian_matrix(Gs).toarray()
        Z_tau_i = cal_Z_tau(L, beta_range)
        nom_Z_tau.append(Z_tau_i/ns)

    return nom_Z_tau