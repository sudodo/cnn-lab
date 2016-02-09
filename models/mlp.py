import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

xp = cuda.cupy if cuda.available else np


class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10),
        )
        if cuda.available:
            cuda.get_device(0).use()
            self.to_gpu()

    def forward(self, x, train):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
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
        return 'mlp'

