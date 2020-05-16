""""demonstarting importance sampling
"""
import numpy as np
import matplotlib.pyplot as plt

NUM_OF_PARTICLES = 50


def main():
    """main method"""
    sigma = 4
    mu = 0
    lim = sigma * 4
    t = np.linspace(-lim, lim, num=NUM_OF_PARTICLES)
    proposal_distribution = np.exp(-0.5 * (t - mu) ** 2 / (sigma ** 2))
    samples = np.random.normal(mu, sigma, NUM_OF_PARTICLES)
    weights = np.ones(NUM_OF_PARTICLES)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, proposal_distribution)
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.stem(samples, weights, use_line_collection='True', markerfmt='None')
    plt.xlim((-lim, lim))
    plt.show()


if __name__ == '__main__':
    main()
