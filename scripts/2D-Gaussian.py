import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def evaluate_gaussian(grid, mu, cov):
    ''' function to calculate a 2D gaussian
    Parameters:
        grid (np.array): grid to evaluate gaussian
        mu (np.array): mean vector
        cov (np.array): covariance matrix
    Returns:
        prob (np.array): 2D gaussian evaluated over the grid '''
    eta = np.sqrt(np.linalg.det(2*np.pi*cov))
    cov_inv = np.linalg.inv(cov)
    z_norm = np.einsum('...k,kl,...l->...', grid - mu, cov_inv, grid - mu)
    prob = np.exp(-0.5*z_norm) / eta
    return prob


def main():
    ''' main method '''
    num_of_grid_points = 101
    cov = np.array([[10, -5], [-5, 10]], dtype=np.float)
    mu = np.array([0, 0], dtype=np.float)
    x_axis = np.linspace(-20, 20, num_of_grid_points)
    y_axis = np.linspace(-20, 20, num_of_grid_points)
    x_values, y_values = np.meshgrid(x_axis, y_axis)
    grid = np.empty(x_values.shape + (2,))
    grid[:, :, 0] = x_values
    grid[:, :, 1] = y_values
    gaussian = evaluate_gaussian(grid, mu, cov)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax1.plot_surface(x_values, y_values, gaussian, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap='viridis')
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    ax2.contourf(x_values, y_values, gaussian, zdir='z', cmap='viridis')
    ax2.view_init(90, 270)
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    plt.show()


if __name__ == '__main__':
    main()
