from sklearn.datasets import fetch_mldata
import numpy as np


class MNIST:
    def __init__(self):
        self.mnist = fetch_mldata('MNIST original', data_home=".")
        np.random.seed(1)

    @staticmethod
    def unison_shuffle(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def _fold_i_split(self, n, i, x, t):
        cls_num = 10
        test_size = len(t)/n
        cls_test_size = test_size/cls_num
        cls_size = len(t)/cls_num
        x_train = x_test = t_train = t_test = None
        for j in range(cls_num):
            cls_head = cls_size*j
            cls_tail = cls_size*(j+1)
            head_idx = cls_head + cls_test_size*i
            tail_idx = cls_head + cls_test_size*(i+1)
            if j == 0:
                x_test = x[head_idx:tail_idx]
                t_test = t[head_idx:tail_idx]
                x_train = np.concatenate((x[cls_head:head_idx], x[tail_idx:cls_tail]))
                t_train = np.concatenate((t[cls_head:head_idx], t[tail_idx:cls_tail]))
            else:
                x_test = np.concatenate((x_test, x[head_idx:tail_idx]))
                t_test = np.concatenate((t_test, t[head_idx:tail_idx]))
                x_train = np.concatenate((x_train, x[cls_head:head_idx], x[tail_idx:cls_tail]))
                t_train = np.concatenate((t_train, t[cls_head:head_idx], t[tail_idx:cls_tail]))
        x_test, t_test = self.unison_shuffle(x_test, t_test)
        x_train, t_train = self.unison_shuffle(x_train, t_train)
        x_train = x_train.reshape((len(x_train), 1, 28, 28))
        x_test = x_test.reshape((len(x_test), 1, 28, 28))
        return x_train, x_test, t_train, t_test

    def get_fold_i(self, n, i):
        x = self.mnist.data
        t = self.mnist.target
        x = x.astype(np.float32)
        t = t.astype(np.int32)
        x /= x.max()
        return self._fold_i_split(n, i, x, t)

