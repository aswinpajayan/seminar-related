""""demonstarting importance sampling
"""
import numpy as np
import matplotlib.pyplot as plt

NUM_OF_PARTICLES = 100


def main():
    """main method"""
    sigma = 4
    mean = 0
    lim = sigma * 4
    t = np.linspace(-lim, lim, num=NUM_OF_PARTICLES)
    proposal_distribution = get_normal(mean, sigma, t)
    samples = np.random.normal(mean, sigma, NUM_OF_PARTICLES)
    weights = np.ones(NUM_OF_PARTICLES) / NUM_OF_PARTICLES
    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(ncols=3, nrows=2,
                            width_ratios=[3, 3, 3], height_ratios=[5, 1])
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.plot(t, proposal_distribution)
    ax1.set_title('proposal distribution')
    plt.xlim((-lim, lim))
    ax2 = fig.add_subplot(spec[1, 0])
    ax2.stem(samples, weights, use_line_collection='True',
             markerfmt='None', linefmt=None)
    plt.xlim((-lim, lim))
    ax2.set_title('samples')

    # creating an arbitrary target distribution
    seed = np.random.random_integers(20, size=(10)) - 10
    print(seed)
    curve = np.zeros(NUM_OF_PARTICLES)
    for p in seed:
        curve += np.random.random_sample() * get_normal(p, np.random.random_integers(5), t)
    target_distribution = curve / np.max(curve)

    index = list(map(lambda sample: np.abs(t - sample).argmin(), samples))
    weights = target_distribution[index] / proposal_distribution[index]
    weights = weights / weights.sum()

    ax3 = fig.add_subplot(spec[0, 1])
    ax3.plot(t, target_distribution)
    ax3.plot(t, proposal_distribution)
    ax3.set_title('target distribution')
    plt.xlim((-lim, lim))
    ax2 = fig.add_subplot(spec[1, 1])
    ax2.stem(samples, weights, use_line_collection='True',
             markerfmt='None', linefmt=None)
    plt.xlim((-lim, lim))
    ax2.set_title('samples')

    # resampling with weights
    rng = np.random.default_rng()
    samples = rng.choice(samples, NUM_OF_PARTICLES, p=weights)
    weights = np.ones(NUM_OF_PARTICLES)/NUM_OF_PARTICLES
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.plot(t, target_distribution)
    ax3.set_title('target distribution')
    plt.xlim((-lim, lim))
    ax2 = fig.add_subplot(spec[1, 2])
    ax2.stem(samples, weights, use_line_collection='True',
             markerfmt='None', linefmt=None)
    plt.xlim((-lim, lim))
    ax2.set_title('after resampling')
    plt.show()


def get_normal(mean, sigma, points):
    curve = np.exp(-0.5 * (points - mean) ** 2 / (sigma ** 2))
    return curve


if __name__ == '__main__':
    main()
