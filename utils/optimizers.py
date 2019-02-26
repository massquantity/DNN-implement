import numpy as np

class Sgd:
    def __init__(self, lr, batch_size):
        self.lr = lr
        self.batch_size = batch_size

    def update(self, weights, biases, grad_w, grad_b):
        for i, (w, gw) in enumerate(zip(weights, grad_w)):
            weights[i] = w - self.lr * gw / self.batch_size

        for i, (b, gb) in enumerate(zip(biases, grad_b)):
            biases[i] = b - self.lr * gb / self.batch_size


class Momentum:
    def __init__(self, lr, momentum, batch_size):
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.velocity = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.velocity is None:
            self.velocity = dict()
            self.velocity['w'] = [np.zeros(w.shape) for w in weights]
            self.velocity['b'] = [np.zeros(b.shape) for b in biases]
        '''
        self.velocity['w'] = [self.momentum * vw - self.lr * gw / self.batch_size
                              for vw, gw in zip(self.velocity['w'], grad_w)]
        self.velocity['b'] = [self.momentum * vb - self.lr * gb / self.batch_size
                              for vb, gb in zip(self.velocity['b'], grad_b)]

        for i, (w, vw) in enumerate(zip(weights, self.velocity['w'])):
            weights[i] = w + vw

        for i, (b, vb) in enumerate(zip(biases, self.velocity['b'])):
            biases[i] = b + vb
        '''

        for i, (w, vw, gw) in enumerate(zip(weights, self.velocity['w'], grad_w)):
            vw = self.momentum * vw - self.lr * gw / self.batch_size
            weights[i] = w + vw
            self.velocity['w'][i] = vw

        for i, (b, vb, gb) in enumerate(zip(biases, self.velocity['b'], grad_b)):
            vb = self.momentum * vb - self.lr * gb / self.batch_size
            biases[i] = b + vb
            self.velocity['b'][i] = vb


class NesterovMomentum:
    def __init__(self, lr, momentum, batch_size):
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.velocity = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.velocity is None:
            self.velocity = dict()
            self.velocity['w'] = [np.zeros(w.shape) for w in weights]
            self.velocity['b'] = [np.zeros(b.shape) for b in biases]

        for i, (w, vw, gw) in enumerate(zip(weights, self.velocity['w'], grad_w)):
            vw = self.momentum * vw - self.lr * gw / self.batch_size
            weights[i] = w + self.momentum * vw - self.lr * gw / self.batch_size
            self.velocity['w'][i] = vw

        for i, (b, vb, gb) in enumerate(zip(biases, self.velocity['b'], grad_b)):
            vb = self.momentum * vb - self.lr * gb / self.batch_size
            biases[i] = b + self.momentum * vb - self.lr * gb / self.batch_size
            self.velocity['b'][i] = vb


class Adam:
    def __init__(self, lr, batch_size, rho1=0.9, rho2=0.999):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.batch_size = batch_size
        self.iteration = 0
        self.ps = None

    def update(self, weights, biases, grad_w, grad_b):
        self.iteration += 1
        grad_w = [gw / self.batch_size for gw in grad_w]
        grad_b = [gb / self.batch_size for gb in grad_b]

        if self.ps is None:
            self.ps = {}
            self.ps['vw'] = [np.zeros(w.shape) for w in weights]
            self.ps['vb'] = [np.zeros(b.shape) for b in biases]
            self.ps['hw'] = [np.zeros(w.shape) for w in weights]
            self.ps['hb'] = [np.zeros(b.shape) for b in biases]

        self.ps['vw'] = [self.rho1 * vw + (1 - self.rho1) * gw for vw, gw in zip(self.ps['vw'], grad_w)]
        self.ps['vb'] = [self.rho1 * vb + (1 - self.rho1) * gb for vb, gb in zip(self.ps['vb'], grad_b)]
        self.ps['hw'] = [self.rho2 * hw + (1 - self.rho2) * (gw ** 2) for hw, gw in zip(self.ps['hw'], grad_w)]
        self.ps['hb'] = [self.rho2 * hb + (1 - self.rho2) * (gb ** 2) for hb, gb in zip(self.ps['hb'], grad_b)]
        unbias_vw = [vw / (1 - self.rho1 ** self.iteration) for vw in self.ps['vw']]
        unbias_vb = [vb / (1 - self.rho1 ** self.iteration) for vb in self.ps['vb']]
        unbias_hw = [hw / (1 - self.rho2 ** self.iteration) for hw in self.ps['hw']]
        unbias_hb = [hb / (1 - self.rho2 ** self.iteration) for hb in self.ps['hb']]

        for i, (w, vw, hw) in enumerate(zip(weights, unbias_vw, unbias_hw)):
            weights[i] = w - self.lr * vw / (np.sqrt(hw) + 1e-8)

        for i, (b, vb, hb) in enumerate(zip(biases, unbias_vb, unbias_hb)):
            biases[i] = b - self.lr * vb / (np.sqrt(hb) + 1e-8)