import sys
sys.path.append('..')
from utils import *


if __name__ == '__main__':
    filename = ["Blogs", "Metabolic", "Drosophila", "Music", "Airports", "Proteome",\
                "USpowergrid", "Gnutella", "Words", "DBLP", "Internet", "Enron"]
    tmin, tmax = 0.0, 6.0
    t = np.linspace(tmin, tmax, 51)

    for net_label in range(2):
        print(filename[net_label])
        G = load_graph_data(filename[net_label])
        N = len(G)
        print(G)

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
        sequence_DC_plus = dict(sorted(DC_plus.items(), key=lambda x: x[1], reverse=False))


        # 不剪枝处理
        Fr, Fi = cal_SIR_rt_it_vs_q(G, sequence_DC_plus, N, q) 
        # 将t和Fr的转置列合并
        data_rt = np.column_stack([t, *Fr])
        np.savetxt("./SIR_Rt_It/SIR_Rt_"+filename[net_label]+ "_DC_plus.dat", data_rt, delimiter=' ', fmt='%g')


        data_it = np.column_stack([t, *Fi])
        np.savetxt("./SIR_Rt_It/SIR_It_"+filename[net_label]+ "_DC_plus.dat", data_it, delimiter=' ', fmt='%g')


        kmin = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        # 剪枝处理
        Fr_prune, Fi_prune = cal_SIR_rt_it_vs_q(G, sequence_DC_plus, N, q[1:], kmin[net_label], prune_flag=True)

        data_rt_p = np.column_stack([t, *Fr_prune])
        np.savetxt("./SIR_Rt_It/SIR_Rt_"+filename[net_label]+ "_DC_plus_prune.dat", data_rt_p, delimiter=' ', fmt='%g')

        data_it_p = np.column_stack([t, *Fi_prune])
        np.savetxt("./SIR_Rt_It/SIR_It_"+filename[net_label]+ "_DC_plus_prune.dat", data_it_p, delimiter=' ', fmt='%g')


