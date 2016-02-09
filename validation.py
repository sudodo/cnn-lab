import numpy as np
from chainer import optimizers
from models import *
from data import MNIST
import copy


def fold_i(model, optimizer, x_train, x_test, t_train, t_test, batch, n_epoch):
    train_size = t_train.size
    test_size = t_test.size
    print train_size, test_size

    optimizer.setup(model)
    sum_accuracies = []
    sum_accuracies_train = []
    sum_losses = []
    for epoch in range(1, n_epoch + 1):
        perm = np.random.permutation(train_size)
        sum_loss = 0
        sum_accuracy_train = 0
        for i in range(0, train_size, batch):
            x_batch = x_train[perm[i:i + batch]]
            y_batch = t_train[perm[i:i + batch]]
            optimizer.update(model, x_batch, y_batch)
            sum_loss += float(model.loss) * len(y_batch)
            sum_accuracy_train += float(model.accuracy) * len(y_batch)

        sum_losses.append(sum_loss / train_size)
        sum_accuracies_train.append(sum_accuracy_train/train_size)
        print 'loss:'+str(sum_loss/train_size)

        sum_accuracy = 0
        for i in range(0, test_size, batch):
            x_batch = x_test[i:i + batch]
            y_batch = t_test[i:i + batch]
            model(x_batch, y_batch, train=False)
            sum_accuracy += float(model.accuracy) * len(y_batch)
        print 'accuracy:'+str(sum_accuracy/test_size)

        sum_accuracies.append(sum_accuracy / test_size)
    print "train mean loss: %f" % (sum_losses[n_epoch-1])
    print "test accuracy: %f" % (sum_accuracies[n_epoch-1])
    return sum_losses, sum_accuracies_train, sum_accuracies


def save_result(path, data):
    data = np.array(data)
    data = np.mean(data, axis=0)
    np.savetxt(path, data)


def k_fold_validation(k, model, tag='', optimizer=optimizers.Adam(), batch=100, n_epoch=100):
    loss = []
    acc = []
    acc_train = []
    mnist = MNIST()
    print str(model)
    for i in range(k):
        x_train, x_test, t_train, t_test = mnist.get_fold_i(k, i)
        print 'fold:' + str(i)
        l, at, a = fold_i(copy.deepcopy(model),
                          copy.deepcopy(optimizer),
                          x_train, x_test, t_train, t_test,
                          batch, n_epoch)
        loss.append(l)
        acc.append(a)
        acc_train.append(at)
    save_result('results/'+str(model)+tag+'_loss.txt', loss)
    save_result('results/'+str(model)+tag+'_accuracy.txt', acc)
    save_result('results/'+str(model)+tag+'_accuracy_train.txt', acc_train)

if __name__ == '__main__':
    #k_fold_validation(10, MLP())
    k_fold_validation(10, CNN())

