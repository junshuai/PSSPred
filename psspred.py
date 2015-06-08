#!/usr/bin/config python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import click
import datetime
import logging
import numpy as np
import theano
import theano.tensor as T

from itertools import product

try:
    import configparser
    import pickle
except ImportError:
    import ConfigParser as configparser
    import cPickle as pickle
    from itertools import izip as zip
    input = raw_input
    range = xrange


__version__ = 0.2


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def piecewise_scaling_func(x):
    if x < -5:
        y = 0.0
    elif -5 <= x <= 5:
        y = 0.5 + 0.1*x
    else:
        y = 1.0
    return y


def encode_residue(residue):
    return [1 if residue == amino_acid else 0
            for amino_acid in ('A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H',
                               'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                               'Y', 'V')]


def encode_dssp(dssp):
    return [1 if dssp == hec else 0 for hec in ('H', 'E', 'C')]


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(floatX(data_x), borrow=borrow)
    shared_y = theano.shared(floatX(data_y), borrow=borrow)
    return shared_x, shared_y


def load_data(filename, window_size=19):
    logging.info('... loading data ("%s")' % filename)

    X = []
    Y = []
    with open(filename, 'r') as f:
        line = f.read().strip().split('\n')
        num_proteins = len(line) // 2

        for line_num in range(num_proteins):
            sequence = line[line_num*2]
            structure = line[line_num*2 + 1]

            double_end = [None] * (window_size // 2)
            unary_sequence = []
            for residue in double_end + list(sequence) + double_end:
                unary_sequence += encode_residue(residue)

            X += [
                unary_sequence[start: start+window_size*20]
                for start in range(0, len(sequence)*20, 20)
            ]

            Y += [encode_dssp(dssp) for dssp in structure]

    return shared_dataset([X, Y])


def load_pssm(filename, window_size=19, scale=piecewise_scaling_func):
    logging.info('... loading pssm ("%s")', filename)

    X = []
    Y = []
    with open(filename, 'r') as f:
        num_proteins = int(f.readline().strip())
        for __ in range(num_proteins):
            m = int(f.readline().strip())
            sequences = []
            for __ in range(m):
                line = f.readline()
                sequences += [scale(float(line[i*3: i*3+3]))
                              for i in range(20)]

            double_end = ([0.]*20) * (window_size//2)
            sequences = double_end + sequences + double_end
            X += [
                sequences[start:start+window_size*20]
                for start in range(0, m*20, 20)
            ]

            structure = f.readline().strip()
            Y += [encode_dssp(dssp) for dssp in structure]

    return shared_dataset([X, Y])


class AccuracyTable(object):

    def __init__(self, pred=None, obs=None):
        self.A = np.zeros(shape=(3, 3), dtype=float)
        if pred is not None and obs is not None:
            self.count(pred, obs)

    """
    Αij = number of residues predicted to be in structure type j and observed
    to be in type i.
    """
    def count(self, pred, obs):
        for p, o in zip(pred, obs):
            self.A[o][p] += 1

    @property
    def Q3(self):
        return self.A.trace() / self.A.sum() * 100

    def C(self, i):
        if not 0 <= i < 3:
            raise ValueError('the argument i can only be 0(helix), 1(strand),'
                             '2(coil)')

        p = self.A[i][i]
        n = sum(self.A[j][k] if j != i and k != i else 0
                for j, k in product(range(3), repeat=2))
        o = sum(self.A[j][i] if j != i else 0 for j in range(3))
        u = sum(self.A[i][j] if j != i else 0 for j in range(3))
        return (p*n-o*u) / ((p+o)*(p+u)*(n+o)*(n+u))**0.5

    @property
    def Ch(self):
        return self.C(0)

    @property
    def Ce(self):
        return self.C(1)

    @property
    def Cc(self):
        return self.C(2)

    @property
    def C3(self):
        return (self.Ch * self.Ce * self.Cc) ** (1./3)

    def __str__(self):
        res = ''
        for i in range(3):
            for j in range(3):
                res += str(self.A[i][j]) + '\t'
            res += '\n'
        return res


class StoppingCriteria(object):
    def __init__(self, k=5):
        self.t = 0
        self.k = k
        self.E_tr = [np.inf]
        self.E_va = [np.inf]
        self.E_opt = np.inf

    def append(self, E_tr, E_va):
        self.t += 1
        self.E_tr.append(E_tr)
        self.E_va.append(E_va)
        self.E_opt = min(self.E_opt, E_va)

    @property
    def generalization_loss(self):
        return 100. * (self.E_va[-1]/self.E_opt - 1)

    @property
    def training_progress(self):
        return 1000. * (sum(self.E_tr[-self.k:]) /
                        (self.k * min(self.E_tr[-self.k:])) - 1)

    def GL(self, alpha):
        """Stop as soon as the generalization loss exceeds a certain threshold.
        """
        return self.generalization_loss > alpha

    def PQ(self, alpha):
        """Stop as soon as quotient of generalization loss and progress exceeds
        a certain threshold
        """
        return self.generalization_loss / self.training_progress > alpha

    def UP(self, s, t=0):
        """Stop when the generalization error increased in s successive strips.
        """
        if t == 0:
            t = self.t
        if t - self.k < 0 or self.E_va[t] <= self.E_va[t - self.k]:
            return False
        if s == 1:
            return True
        return self.UP(s - 1, t - self.k)


def init_weights_sigmoid(shape):
    low = -np.sqrt(6./(shape[0]+shape[1])) * 4.
    high = np.sqrt(6./(shape[0]+shape[1])) * 4.
    values = np.random.uniform(low=low, high=high, size=shape)
    return theano.shared(floatX(values), borrow=True)


def init_weights(shape):
    values = np.random.randn(*shape)*0.01
    return theano.shared(floatX(values), borrow=True)


def init_bias(shape):
    values = np.zeros(shape, dtype=theano.config.floatX)
    return theano.shared(values, borrow=True)


class MultilayerPerceptron(object):
    def __init__(self, n_input, n_hidden, n_output):
        logging.info('... building model (%d-%d-%d)',
                     n_input, n_hidden, n_output)

        self.W_h = init_weights_sigmoid((n_input, n_hidden))
        self.b_h = init_bias(n_hidden)
        self.W_o = init_weights((n_hidden, n_output))
        self.b_o = init_bias(n_output)

        self.params = [self.W_h, self.b_h, self.W_o, self.b_o]

        self.X = T.matrix()
        self.Y = T.matrix()

        h = T.nnet.sigmoid(T.dot(self.X, self.W_h) + self.b_h)
        self.py_x = T.nnet.softmax(T.dot(h, self.W_o) + self.b_o)

        y = T.argmax(self.Y, axis=1)
        self.NLL = -T.mean(T.log(self.py_x)[T.arange(self.Y.shape[0]), y])
        self.L1 = T.sum(abs(self.W_h)) + T.sum(abs(self.W_o))
        self.L2_sqr = T.sum((self.W_h**2)) + T.sum((self.W_o**2))

    def train_model(self, X_train, Y_train, X_valid, Y_valid,
                    num_epochs=3000, learning_rate=0.001, batch_size=20,
                    L1_reg=0., L2_reg=0.):

        logging.info('... training model (learning_rate: %f)' % learning_rate)

        cost = self.NLL + L1_reg*self.L1 + L2_reg*self.L2_sqr

        grads = T.grad(cost=cost, wrt=self.params)
        updates = [[param, param - learning_rate*grad]
                   for param, grad in zip(self.params, grads)]

        start = T.lscalar()
        end = T.lscalar()

        train = theano.function(
            inputs=[start, end],
            outputs=cost,
            updates=updates,
            givens={
                self.X: X_train[start:end],
                self.Y: Y_train[start:end]
            }
        )

        validate = theano.function(
            inputs=[start, end],
            outputs=[cost, self.py_x],
            givens={
                self.X: X_valid[start:end],
                self.Y: Y_valid[start:end]
            }
        )

        m_train = X_train.get_value(borrow=True).shape[0]
        m_valid = X_valid.get_value(borrow=True).shape[0]

        stopping_criteria = StoppingCriteria()
        index = range(0, m_train+1, batch_size)

        y_valid = np.argmax(Y_valid.get_value(borrow=True), axis=1)
        for i in range(num_epochs):
            costs = [train(index[j], index[j+1]) for j in range(len(index)-1)]
            E_tr = np.mean(costs)

            E_va, py_x = validate(0, m_valid)
            y_pred = np.argmax(py_x, axis=1)
            A_valid = AccuracyTable(y_pred, y_valid)

            stopping_criteria.append(E_tr, E_va)
            logging.debug('epoch %3d/%d. Cost: %f  Validation: Q3=%.2f%% C3=%f'
                          '(%.2f %.2f %.2f)',
                          i+1, num_epochs, E_tr, A_valid.Q3, A_valid.C3,
                          A_valid.Ch, A_valid.Ce, A_valid.Cc)

            if stopping_criteria.PQ(1):
                logging.debug('Early Stopping!')
                break

        return stopping_criteria

    def predict(self, X):
        start = T.lscalar()
        end = T.lscalar()
        return theano.function(
            inputs=[start, end],
            outputs=self.py_x,
            givens={self.X: X[start:end]}
        )


class Config(object):

    def __init__(self, profile, section):
        parser = configparser.RawConfigParser()
        parser.read(profile)

        self.train_file = parser.get(section, 'training_file')
        self.valid_file = parser.get(section, 'validation_file')
        self.window_size = parser.getint(section, 'window_size')
        self.hidden_layer_size = parser.getint(section, 'hidden_layer_size')
        self.learning_rate = parser.getfloat(section, 'learning_rate')
        self.num_epochs = parser.getint(section, 'num_epochs')
        if section == 'SECOND':
            self.network_file = parser.get(section, 'network_file')


def first_level(cfg, target):
    now = datetime.datetime.now()
    logging.info(now.strftime('%Y-%m-%d %H:%M:%S'))

    X_train, Y_train = load_pssm(cfg.train_file, window_size=cfg.window_size)
    X_valid, Y_valid = load_pssm(cfg.valid_file, window_size=cfg.window_size)

    input_layer_size = cfg.window_size * 20
    output_layer_size = 3

    classifier = MultilayerPerceptron(input_layer_size,
                                      cfg.hidden_layer_size,
                                      output_layer_size)

    result = classifier.train_model(X_train, Y_train, X_valid, Y_valid,
                                    cfg.num_epochs,
                                    learning_rate=cfg.learning_rate)

    if target is not None:
        for E_tr, E_va in zip(result.E_tr, result.E_va):
            target.write(str(E_tr) + ',' + str(E_va) + '\n')

    network_file = now.strftime('%Y%m%dT%H%M%S') + '.nn1'
    logging.info('... saving model in file (%s)', network_file)
    pickle.dump(classifier, open('output/' + network_file, 'wb'))

    return classifier


def second_level(cfg, fst_layer_classifier, target):

    def transform(x, m, window_size=17):
        double_end = [0.] * 3 * (window_size // 2)
        sequences = double_end + x.tolist()[0] + double_end
        return [sequences[index: index+window_size*3]
                for index in range(0, m*3, 3)]

    def get_XY(filename):
        X_data, Y_data = load_pssm(filename)
        m = X_data.get_value(borrow=True).shape[0]
        predict = fst_layer_classifier.predict(X_data)
        x = predict(0, m).reshape(1, m*3)
        x = transform(x, m, cfg.window_size)
        X = theano.shared(floatX(x), borrow=True)
        return X, Y_data

    now = datetime.datetime.now()
    logging.info(now.strftime('%Y-%m-%d %H:%M:%S'))

    if fst_layer_classifier is None:
        with open(cfg.network_file, 'rb') as f:
            fst_layer_classifier = pickle.load(f)

    X_train, Y_train = get_XY(cfg.train_file)
    X_valid, Y_valid = get_XY(cfg.valid_file)

    snd_layer_classifier = MultilayerPerceptron(cfg.window_size*3,
                                                cfg.hidden_layer_size,
                                                3)
    result = snd_layer_classifier.train_model(X_train, Y_train,
                                              X_valid, Y_valid,
                                              cfg.num_epochs,
                                              cfg.learning_rate)

    if target is not None:
        for E_tr, E_va in zip(result.E_tr, result.E_va):
            target.write(str(E_tr) + ',' + str(E_va) + '\n')

    network_file = now.strftime('%Y%m%dT%H%M%S') + '.nn2'
    logging.info('... saving model in file (%s)' % network_file)
    pickle.dump(snd_layer_classifier, open('output/' + network_file, 'wb'))


@click.group()
@click.option('--verbose', '-v', is_flag=True,
              help='Show the detail.')
@click.option('--quiet', '-q', is_flag=True,
              help='Do not print any infomation.')
@click.version_option(version=__version__)
def cli(verbose, quiet):
    """Protein Secondary Structure Prediction"""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='', level=level)


@cli.command()
@click.option('--first-only', is_flag=True,
              help='Train the first-level network only.')
@click.option('--second-only', is_flag=True,
              help='Train the second-level network only.')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='The config file to use instead of the default.')
@click.option('--learning-rate', '-lr', default=0.,
              help='Set the learning rate of training manually.')
@click.option('--save-training-progress', '-s', 'target', type=click.File('w'),
              help='Save the training progress into a file')
def train(first_only, second_only, config, learning_rate, target):
    if config is None:
        config = 'default.cfg'

    config4first = Config(config, 'FIRST')
    config4second = Config(config, 'SECOND')

    if learning_rate > 0:
        config4first.learning_rate = learning_rate
        config4second.learning_rate = learning_rate

    classifier = None
    if not second_only:
        classifier = first_level(config4first, target)
    if not first_only:
        second_level(config4second, classifier, target)


@cli.command()
@click.option('--first', '-1', type=click.Path(exists=True),
              help='The first-level network file.')
@click.option('--second', '-2', type=click.Path(exists=True),
              help='The second-level network file.')
@click.argument('filename', type=click.Path(exists=True))
def test(first, second, filename):

    def transform(x, m, window_size=17):
        double_end = [0.] * 3 * (window_size // 2)
        sequences = double_end + x.tolist()[0] + double_end
        return [sequences[index: index+window_size*3]
                for index in range(0, m*3, 3)]

    def get_XY(window_size=17):
        X_data, Y_data = load_pssm(filename)
        m = X_data.get_value(borrow=True).shape[0]
        predict = fst_layer_classifier.predict(X_data)
        x = predict(0, m).reshape(1, m*3)
        x = transform(x, m, window_size)
        X = theano.shared(floatX(x), borrow=True)
        return X, Y_data

    now = datetime.datetime.now()
    logging.info(now.strftime('%Y-%m-%d %H:%M:%S'))

    with open(first, 'rb') as f:
        fst_layer_classifier = pickle.load(f)

    with open(second, 'rb') as f:
        snd_layer_classifier = pickle.load(f)

    window_size = snd_layer_classifier.W_h.get_value(borrow=True).shape[0] // 3
    X_test, Y_test = get_XY(window_size)

    snd_layer_classifier.train_model(X_test, Y_test, X_test, Y_test, 1)
