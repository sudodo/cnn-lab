import matplotlib.pyplot as plt
import numpy as np


def averaging(seq, sigma=0):
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


def make_graph_cmp(title, sigma, *kwargs):

    std = []
    max = []
    for path, label in kwargs:
        acc = np.loadtxt(path)
        std.append([label, np.std(acc)])
#        print label + ": " + str(np.std(acc))
        acc = averaging(acc, sigma)
        max.append([label, np.max(acc)])
        print label + ": " + str(np.max(acc))
        plt.plot(acc, label=label)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0.98, 0.994])
    plt.legend(loc='lower right')
    plt.title(title)
    plt.grid()
    plt.show()

    plt.bar(range(len(std)), map(float, np.array(std)[:, 1].tolist()), align='center')
    plt.xticks(range(len(std)), np.array(std)[:, 0])
    plt.title('standard deviation')
    plt.show()

    plt.bar(range(len(max)), map(float, np.array(max)[:, 1].tolist()), align='center')
    plt.xticks(range(len(max)), np.array(max)[:, 0])
    plt.title('max accuracy')
    plt.ylim([0.985, 0.994])
    plt.show()

def best_optimizer():
    make_graph_cmp(
        "which is the best optimizer[sigma=0]",
        0,
        ('results/cnn_SGD_accuracy.txt', 'SGD'),
        ('results/cnn_MomentumSGD_accuracy.txt', 'MomentumSGD'),
        ('results/cnn_NesterovAG_accuracy.txt', 'NesterovAG'),
        ('results/cnn_RMSpropGraves_accuracy.txt', 'RMSpropGraves'),
        ('results/cnn_AdaDelta_accuracy.txt', 'AdaDelta'),
        ('results/cnn_AdaGrad_accuracy.txt', 'AdaGrad'),
        ('results/cnn_Adam_accuracy.txt', 'Adam')
                   )


if __name__ == '__main__':
    best_optimizer()
#    make_graph('results/mlp_accuracy.txt', 'results/mlp_accuracy_train.txt')

