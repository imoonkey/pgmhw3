__author__ = 'moonkey'

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# P 1.4.1
def generate_data(count):
    data = [
        (np.random.randn() - 5) if (np.random.rand() > 0.5) else (np.random.randn() + 5)
        for _ in range(count)]
    # plt.plot(data)
    # plt.show()
    return np.array(data)


def norm_pdf(x, mu, sigma):
    # x = (x - mu) / sigma
    # p = np.exp(-0.5 * (x ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return norm.pdf(x, mu, sigma)


def p_tilde(mu, xs):
    prior = np.exp(-0.005 * (mu[0] ** 2) + -0.005 * (mu[1] ** 2))
    likelihood = [
        (0.5 * np.exp(-0.5 * (x - mu[0]) ** 2) + 0.5 * np.exp(-0.5 * (x - mu[1]) ** 2))
        for x in xs]
    total_likelihood = np.prod(likelihood)
    return prior * total_likelihood


# P 1.4.2
def metropolis_hastings(xs, burn=10000, use=1000, sigma=0.5, exp_postfix=''):
    def sample_proposed_q(mu):
        mu_p0 = np.random.randn() * sigma + mu[0]
        mu_p1 = np.random.randn() * sigma + mu[1]
        return [mu_p0, mu_p1]

    def proposed_q(mu_p, mu):  # q(mu_p|mu)
        res = norm_pdf(mu_p[0], mu[0], sigma)
        res *= norm_pdf(mu_p[1], mu[1], sigma)
        return res

    accepted_count = 0
    samples = []
    mu_t = [0, 0]
    for iter in range(burn + use):
        mu_p = sample_proposed_q(mu_t)
        # temp = proposed_q(mu_t, mu_p) / proposed_q(mu_p, mu_t)
        # assert abs(temp - 1) < 1e-5
        temp = 1
        temp *= p_tilde(mu_p, xs) / p_tilde(mu_t, xs)
        acc = min(1, temp)
        if np.random.rand() < acc:
            accepted_count += 1
            mu_t = mu_p
        if iter >= burn:
            samples.append(mu_t)
    acc_rate = accepted_count / float(burn + use)
    mu_mean = np.array(samples).mean(axis=0)
    plt.style.use('ggplot')
    plt.scatter([s[0] for s in samples], [s[1] for s in samples])
    plt.xlabel('$\mu_1$')
    plt.ylabel('$\mu_2$')
    plt.savefig('img/mh_' + exp_postfix + '.pdf')
    plt.clf()
    return acc_rate, mu_mean


# P 1.4.3
def gibbs(xs, burn=10000, use=1000, exp_postfix=''):
    def prob_z_zero(x, mu):  # p(z=0|x, mu)
        p0 = norm_pdf(x, mu[0], 1)
        p1 = norm_pdf(x, mu[1], 1)
        return p0 / (p0 + p1)

    def sample_mu_posterior(x0sum, x0count):
        posterior_mean = (np.sum(x0sum)) / (0.01 + x0count)
        posterior_sigma = 1 / np.sqrt(0.01 + x0count)
        print posterior_mean, posterior_sigma
        return np.random.randn() * posterior_sigma + posterior_mean

    samples = []
    mu_t = [0, 0]
    zs = [0 for _ in xrange(len(xs))]
    for iter in range(burn + use):
        # sample zs given xs and mu
        zs = np.array(np.random.rand(len(zs)) > prob_z_zero(xs, mu_t), dtype=int)
        # zs = [0 if np.random.rand() < prob_z_zero(xs[z_idx], mu_t) else 1
        # for z_idx in range(len(zs))]
        # sample mu
        x1sum = np.dot(xs, zs)
        x1count = np.sum(zs)
        x0sum = np.sum(xs) - x1sum
        x0count = len(zs) - x1count
        mu_t = [sample_mu_posterior(x0sum, x0count), sample_mu_posterior(x1sum, x1count)]
        if iter >= burn:
            samples.append(mu_t)
    mu_mean = np.array(samples).mean(axis=0)
    plt.style.use('ggplot')
    plt.scatter([s[0] for s in samples], [s[1] for s in samples])
    plt.xlabel('$\mu_1$')
    plt.ylabel('$\mu_2$')
    plt.savefig('img/gibbs_' + exp_postfix + '.pdf')
    plt.clf()
    return mu_mean


def print_table(acc_rates, mu_means, filename):
    table = ''
    for exp_idx in range(len(mu_means)):
        mu_mean = mu_means[exp_idx]
        table += ' & '.join([str(mu_mean[0]), str(mu_mean[1])])
        if acc_rates:
            acc_rate = acc_rates[exp_idx]
            table += ' & ' + str(acc_rate)
        table += '\\\\\n'
    with open(filename, 'w') as table_file:
        table_file.write(table)
    return table


def main():
    # np.random.seed(0)
    exp_times = 1
    data = generate_data(count=100)

    # # MH SAMPLING
    # for sigma in [0.5, 5]:
    # acc_rates = []
    # mu_means = []
    #     for exp_idx in range(exp_times):
    #         acc_rate, mu_mean = metropolis_hastings(
    #             data, burn=10000, use=1000, sigma=sigma, exp_postfix=str(sigma) + '_' + str(exp_idx))
    #         acc_rates.append(acc_rate)
    #         mu_means.append(mu_mean)
    #     print_table(acc_rates, mu_means, 'mh_' + str(sigma) + '.txt')

    # GIBBS SAMPLING
    mu_means = []
    for exp_idx in range(exp_times):
        mu_mean = gibbs(
            data, burn=10000, use=1000, exp_postfix=str(exp_idx))
        mu_means.append(mu_mean)
    print_table(None, mu_means, 'gibbs.txt')


if __name__ == '__main__':
    main()