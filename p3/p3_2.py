__author__ = 'moonkey'

import numpy as np
import logging
import matplotlib.pyplot as plt


def read_data(filename='hw3mmsb/hw3train.data'):
    with open(filename) as data_file:
        num_nodes = int(data_file.readline())
        adj_mtx = np.zeros([num_nodes, num_nodes], dtype=int)
        neighbors = [[] for _ in range(num_nodes)]
        for idx in range(num_nodes):
            line = data_file.readline().rstrip(' \n').split(' ')
            neighbors[idx] = [int(s) for s in line]
            for j in neighbors[idx]:
                adj_mtx[idx, j] = 1
    return adj_mtx


def lld(theta, beta, adj_mtx):
    num_nodes = adj_mtx.shape[0]

    neg_beta = 1 - beta

    def ld_local(i, j):
        y_ij = adj_mtx[i, j]
        temp = beta if y_ij else neg_beta
        temp = np.dot(np.dot(theta[i], temp), theta[j])
        return temp

    ld_locals = np.array([[ld_local(i, j) for i in xrange(num_nodes)] for j in xrange(num_nodes)])
    ld_locals = np.log(ld_locals)
    ld_locals = ld_locals - np.diag(np.diag(ld_locals))
    lld_res = ld_locals.sum().sum()

    # lld_res = 0
    # for i in xrange(num_nodes):
    # for j in xrange(num_nodes):
    # if i == j:
    # continue
    # y_ij = adj_mtx[i, j]
    # temp = beta if y_ij else (1 - beta)
    # temp = np.dot(np.dot(theta[i], temp), theta[j])
    # lld_res += np.log(temp)
    return lld_res


def between_com_count(z_pairs, K):
    # n(p, q) of community p and q
    num_nodes = z_pairs.shape[0]
    flatten_z = z_pairs.reshape([num_nodes * num_nodes, 2])
    n_pq = np.histogram2d(flatten_z[:, 0], flatten_z[:, 1], bins=K, range=[[0, K], [0, K]])[0].astype(dtype=int)
    return n_pq


def gibbs_mmsb(adj_mtx, max_iter=10000, use=1000):
    K = 5
    alpha = 0.02
    eta = [0.01, 0.05]

    lld_list = []
    beta_list = []
    num_nodes = adj_mtx.shape[0]

    # z(i,j) ~(Dis(\theta_i), Dis(\theta_j))
    z_pairs = np.random.randint(low=0, high=K, size=[num_nodes, num_nodes, 2])


    # m_ip = np.zeros([num_nodes, K])
    # for i in range(num_nodes):
    # m_ip[i] = np.bincount(np.concatenate([z_pairs[i, :, 0], z_pairs[:, i, 1]]), minlength=K)
    np.fill_diagonal(z_pairs[:, :, 0], 0)
    np.fill_diagonal(z_pairs[:, :, 1], 0)
    m_ip = np.array([
        np.bincount(np.concatenate([z_pairs[i, :, 0], z_pairs[:, i, 1]]), minlength=K)
        for i in range(num_nodes)
    ])
    m_ip[:, 0] -= 2

    np.fill_diagonal(z_pairs[:, :, 0], -1)
    np.fill_diagonal(z_pairs[:, :, 1], -1)
    n_pq = between_com_count(z_pairs, K)
    pos_link_z_pairs = adj_mtx[:, :, np.newaxis] * z_pairs - (1 - adj_mtx)[:, :, np.newaxis]
    n_pq_pos = between_com_count(pos_link_z_pairs, K)
    n_pq_neg = n_pq - n_pq_pos
    n_pq_sign = np.concatenate([n_pq_neg[:, :, np.newaxis], n_pq_pos[:, :, np.newaxis]], axis=2)
    # neg_link_z_pairs = (1 - adj_mtx)[:, :, np.newaxis] * z_pairs - adj_mtx[:, :, np.newaxis]
    # n_pq_neg = between_com_count(neg_link_z_pairs, K)
    # assert (n_pq == n_pq_pos + n_pq_neg).all()

    # all_pairs = [(i, j) for i in range(0, num_nodes) for j in range(0, num_nodes) if i != j]
    for iter in range(max_iter):
        # for (i, j) in all_pairs:
        for i in xrange(num_nodes):
            for j in xrange(num_nodes):
                # sample z_ij
                if i == j:
                    continue
                y_ij = adj_mtx[i, j]
                z_ij = z_pairs[i, j]
                # subtract link ij, i.e. z_ij from n_pq and m_ip
                n_pq[z_ij[0], z_ij[1]] -= 1
                n_pq_sign[z_ij[0], z_ij[1], y_ij] -= 1
                m_ip[i, z_ij[0]] -= 1
                m_ip[j, z_ij[1]] -= 1

                # cal prob
                n_mtx = (n_pq_sign[:, :, y_ij] + eta[y_ij]) / (n_pq + eta[0] + eta[1])
                left = np.diag(m_ip[i, :] + alpha)
                right = np.diag(m_ip[j, :] + alpha)
                prob_table = np.dot(np.dot(left, n_mtx), right)
                prob_table /= prob_table.sum()

                # sample
                choices = K ** 2
                index = np.random.choice(choices, size=1, p=prob_table.ravel())
                new_k = np.unravel_index(index, dims=[K, K])

                # add z_ij back with the newly sampled z_ij
                z_pairs[i, j] = new_k
                n_pq[new_k[0], new_k[1]] += 1
                n_pq_sign[new_k[0], new_k[1], y_ij] += 1
                m_ip[i, new_k[0]] += 1
                m_ip[j, new_k[1]] += 1

        # estimate \theta and \beta, and compute lld
        theta = m_ip + alpha
        row_sums = theta.sum(axis=1)
        theta = theta / row_sums[:, np.newaxis]
        beta = (n_pq_sign[:, :, 1] + eta[1]) / (n_pq + eta[0] + eta[1])
        beta_list.append(beta)
        beta_list = beta_list[-use:]
        lld_t = lld(theta, beta, adj_mtx)
        lld_list.append(lld_t)
        lld_list = lld_list
        logging.info('Iter:' + str(iter) + ' lld:' + str(lld_t))
        if iter % 1000 == 0:
            np.savez('output', lld_list=lld_list, z=z_pairs, beta_list=beta_list)
    np.savez('output', lld_list=lld_list, z=z_pairs, beta_list=beta_list)


def plot_res():
    output = np.load('output.npz')
    beta_list = output['beta_list']
    lld_list = output['lld_list']
    plt.plot(lld_list)
    plt.show()


def main():
    adj_mtx = read_data()
    gibbs_mmsb(adj_mtx)
    plot_res()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename='p3.log'
    )
    main()
    # profile.run('main()')