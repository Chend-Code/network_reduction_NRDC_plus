import sys
sys.path.append('..')  # 添加上级目录到搜索路径
from utils import *

if __name__ == '__main__':
    filename = ["Blogs", "Metabolic", "Drosophila", "Music", "Airports", "Proteome",\
                "USpowergrid", "Gnutella", "Words", "DBLP", "Internet", "Enron"]

    beta_list = np.linspace(0, 2.0, 41)
    kmin = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    for net_label in range(12):
        print(filename[net_label])
        G = load_graph_data(filename[net_label])
        N = len(G)

        # 设定移除节点比例
        if N < 10000:
            l = 4
        elif N >= 10000 and N < 20000:
            l = 5
        else:
            l = 6

        q = [1- 1./2**i for i in range(l)]
        print(q)

        DC_plus = cal_DC_plus(G)
        sequence = dict(sorted(DC_plus.items(), key=lambda x: x[1], reverse=False))

        # 不剪枝
        rhoR = cal_rho_r_vs_beta_q(G, sequence, N, q, beta_list)
        data = np.column_stack([beta_list, *rhoR])
        np.savetxt("./SIR_rho_vs_beta/SIR_rho_vs_beta_"+filename[net_label] + "_DC_plus.dat", data, delimiter=' ', fmt='%g')

        # 剪枝
        rhoR_prune = cal_rho_r_vs_beta_q(G, sequence, N, q[1:], beta_list, kmin[net_label], prune_flag=True)
        data_prune = np.column_stack([beta_list, *rhoR_prune])
        np.savetxt("./SIR_rho_vs_beta/kmin2/SIR_rho_vs_beta_prune_"+filename[net_label] + "_DC_plus.dat", data_prune, delimiter=' ', fmt='%g')

