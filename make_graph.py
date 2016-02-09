import matplotlib.pyplot as plt
import numpy as np


def averaging(seq, sigma=0.01):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(seq, sigma=sigma)


def make_graph(acc_path, loss_path, xlim=300):
    acc = np.loadtxt(acc_path)
    loss = np.loadtxt(loss_path)
    acc = averaging(acc)
    loss = averaging(loss)
    plt.plot(acc[:xlim], label='test')
    plt.plot(loss[:xlim], label='train')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0.8, 1])
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def make_graph_cmp(*kwargs):
    for path, label in kwargs:
        acc = np.loadtxt(path)
        acc = averaging(acc)
        plt.plot(acc, label=label)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0.8, 1])
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    make_graph('results/mlp_accuracy.txt', 'results/mlp_accuracy_train.txt')
