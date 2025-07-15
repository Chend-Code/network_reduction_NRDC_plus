import sys
sys.path.append('..')  # 添加上级目录到搜索路径
from utils import *


if __name__ == '__main__':
    # 设定移除节点比例
    l = 4
    q = [1- 1./2**i for i in range(l)]
    print(q)

    N = 5000
    M = 5*N
    m = 5

    beta_list = np.linspace(0, 2.0, 41)
    rhoR_ER = np.zeros((4,41))
    rhoR_BA = np.zeros((4,41))
    samples = 10
    for i in range(samples):
        G_ER = nx.gnm_random_graph(N, M)
        DC_plus_ER = cal_DC_plus(G_ER)
        sequence_ER = dict(sorted(DC_plus_ER.items(), key=lambda x: x[1], reverse=False))
        rhoRi_ER = cal_rho_r_vs_beta_q(G_ER, sequence_ER, N, q, beta_list)
        rhoR_ER += np.array(rhoRi_ER).reshape(4, 41)


        G_BA = nx.barabasi_albert_graph(N, m)
        DC_plus_BA = cal_DC_plus(G_BA)
        sequence_BA = dict(sorted(DC_plus_BA.items(), key=lambda x: x[1], reverse=False))
        rhoRi_BA = cal_rho_r_vs_beta_q(G_BA, sequence_BA, N, q, beta_list)
        rhoR_BA += np.array(rhoRi_BA).reshape(4, 41)



    average_rhoR_ER = rhoR_ER/samples
    outf = open("./SIR_rho_vs_beta/SIR_rho_vs_beta_ER_DC_plus.dat", "w")
    for j in range(len(beta_list)):
        outf.write(str(beta_list[j])+" "+str(average_rhoR_ER[0][j])+" "+str(average_rhoR_ER[1][j])+" "+str(average_rhoR_ER[2][j])+" "+str(average_rhoR_ER[3][j])+"\n")
    outf.close()


    average_rhoR_BA = rhoR_BA/samples
    outf = open("./SIR_rho_vs_beta/SIR_rho_vs_beta_BA_DC_plus.dat", "w")
    for j in range(len(beta_list)):
        outf.write(str(beta_list[j])+" "+str(average_rhoR_BA[0][j])+" "+str(average_rhoR_BA[1][j])+" "+str(average_rhoR_BA[2][j])+" "+str(average_rhoR_BA[3][j])+"\n")
    outf.close()

