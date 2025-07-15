import sys
sys.path.append('..')  # 添加上级目录到搜索路径
from utils import *

# Function to calculate the heterogeneity index H
def calculate_H(G):
    N = G.number_of_nodes()
    degrees = np.array([degree for node, degree in G.degree()], dtype=np.int64)
    avg_degree = np.mean(degrees)

    H = np.sum(np.abs(degrees[:, None] - degrees[None, :])) / (2 * N**2 * avg_degree)


    return H




if __name__ == '__main__':
    filename = ["Blogs", "Metabolic", "Drosophila", "Music", "Airports", "Proteome",\
                "USpowergrid", "Gnutella", "Words", "DBLP", "Internet", "Enron"]

    for net_label in range(12):
        G = load_graph_data(filename[net_label])
        N, M = nx.number_of_nodes(G), nx.number_of_edges(G)
        print(filename[net_label])
        print(N, M, 2*M/N)
        print(nx.is_connected(G))


        # if not nx.is_connected(G):
        #     G = get_LCC(G)
        # print(G)

        H = calculate_H(G)
        print(H)
