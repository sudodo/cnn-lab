import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

xp = cuda.cupy if cuda.available else np


class CNN(Chain):
    def __init__(self, channel=1, c1=16, c2=32, c3=64, f1=256, f2=512, filter_size1=3, filter_size2=3, filter_size3=3):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(channel, c1, filter_size1),
            conv2=L.Convolution2D(c1, c2, filter_size2),
            conv3=L.Convolution2D(c2, c3, filter_size3),
            l1=L.Linear(f1, f2),
            l2=L.Linear(f2, 10)
        )
        if cuda.available:
            cuda.get_device(0).use()
            self.to_gpu()

    def forward(self, x, train):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.l1(h)), train=train)
        y = self.l2(h)
        return y

    def __call__(self, x, t, train=True):
        x = Variable(xp.asarray(x))
        t = Variable(xp.asarray(t))
        y = self.forward(x, train=train)
        loss = F.softmax_cross_entropy(y, t)
        self.loss = loss.data
        self.accuracy = F.accuracy(y, t).data
        return loss

    def __str__(self):
        return 'cnn'


