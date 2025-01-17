'''
Author: Jia Wang
'''
import numpy as np
import networkx as nx
import theano
# from theano import tensor
from theano import config
from collections import OrderedDict
import timeit
import six.moves.cPickle as pickle
import downhill
import metrics
# import pdb
import pprint
import json
import data_utils
import tprnn_model
import sys

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def init_params(options):
    """
    Initializes values of shared variables.
    """
    params = OrderedDict()

    # word embedding, shape = (n_words, dim_proj)
    randn = np.random.randn(options['n_words'],
                            options['dim_proj'])
    params['Wemb'] = (0.1 * randn).astype(config.floatX)

    # shape = dim_proj * (4*dim_proj)
    lstm_W = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_W'] = lstm_W

    # shape = dim_proj * (4*dim_proj)
    lstm_U = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_U'] = lstm_U

    lstm_b = np.zeros((4 * options['dim_proj'],))
    params['lstm_b'] = lstm_b.astype(config.floatX)

    # decoding matrix for external influences
    randn = np.random.randn(options['dim_proj'],
                            options['n_words'])
    params['W_ext'] = (0.1 * randn).astype(config.floatX)
    dec_b = np.zeros(options['n_words'])
    params['b_ext'] = dec_b.astype(config.floatX)

    return params


def init_tparams(params):
    '''
    Set up Theano shared variables.
    '''
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def evaluate(f_prob, test_loader, k_list=[10, 50, 100], test_batch=False):
    '''
    Evaluates trained model.
    '''
    n_batches = len(test_loader)
    print("n_batches---",n_batches)
    y = None
    y_prob = None
    y_hate = None
    for _ in range(n_batches):
        batch_data = test_loader()
        y_ = batch_data[-1]
#         print y_
#         print y_h_
#         print len(y_)
#         print len(y_h_)
#         print type(y_)
#         print type(y_h_)
#         if test_batch:
        y_prob_ = f_prob(*batch_data[:-2])
        y_h_ = batch_data[-2]
#         else:
#             y_prob_ = f_prob(*batch_data[:-1])
#             y_h_=None 

        # excludes activated nodes when predicting.
        for i, p in enumerate(y_prob_):
            length = int(np.sum(batch_data[1][:, i]))
            sequence = batch_data[0][: length, i]
            assert y_[i] not in sequence, str(sequence) + str(y_[i])
            p[sequence] = 0.
            y_prob_[i, :] = p / float(np.sum(p))

        if y_prob is None:
            y_prob = y_prob_
            y = y_
            y_hate = y_h_
#         if y_prob is None and test_batch:
#             y_hate = y_h_
        else:
            y = np.concatenate((y, y_), axis=0)
            y_prob = np.concatenate((y_prob, y_prob_), axis=0)
#             if test_batch:
            y_hate = np.concatenate((y_hate, y_h_),axis=0)

    return metrics.portfolio(y_prob, y, y_hate, k_list=k_list, test_batch=test_batch)


def train(data_dir='data/twitter/',
          dim_proj=512,
          maxlen=30,
          batch_size=64,
          keep_ratio=1.,
          shuffle_data=True,
          learning_rate=0.001,
          global_steps=10,
          disp_freq=50,
          save_freq=50,
          test_freq=50,
          saveto_file='params.npz',
          weight_decay=0.0005,
          reload_model=False,
          train=True):
    """
    Topo-LSTM model training.
    """
    options = locals().copy()
    saveto = data_dir + saveto_file

    # loads graph
    G, node_index, hate_labels = data_utils.load_graph(data_dir)
#     print(hate_labels)
#     n_nodes = len(node_index)
    # print("Nnodes", n_nodes)
    print nx.info(G)
    options['n_words'] = len(node_index)
    sys.stdout.flush()
    print options

    # creates and initializes shared variables.
    print 'Initializing variables...'
    params = init_params(options)
    if reload_model:
        print 'reusing saved model.'
        load_params(saveto, params)
    tparams = init_tparams(params)

    # builds Topo-LSTM model
    print 'Building model...'
    model = tprnn_model.build_model(tparams, options)

    print 'Loading test data...'
    test_examples = data_utils.load_examples(data_dir,
                                             dataset='test',
                                             node_index=node_index,
                                             maxlen=maxlen,
                                             hate_labels=hate_labels,
                                             G=G)
#     print test_examples
    test_loader = data_utils.Loader(test_examples, hate=True, options=options)
    print 'Loaded %d test examples' % len(test_examples)
    sys.stdout.flush()
    if train:
        # prepares training data.
        print 'Loading train data...'
        train_examples = data_utils.load_examples(data_dir,
                                                  dataset='train',
                                                  keep_ratio=options[
                                                      'keep_ratio'],
                                                  node_index=node_index,
                                                  maxlen=maxlen,
                                                  hate_labels=None,
                                                  G=G)
        train_loader = data_utils.Loader(train_examples, hate=False, options=options)
        print 'Loaded %d training examples.' % len(train_examples)

        # compiles updates.
        optimizer = downhill.build(algo='adam',
                                   loss=model['cost'],
                                   params=tparams.values(),
                                   inputs=model['data'])

        updates = optimizer.get_updates(max_gradient_elem=5.,
                                        learning_rate=learning_rate)

        f_update = theano.function(model['data'],
                                   model['cost'],
                                   updates=list(updates))

        # training loop.
        start_time = timeit.default_timer()
        sys.stdout.flush()
        # downhill.minimize(
        #     loss=cost,
        #     algo='adam',
        #     train=train_loader,
        #     # inputs=input_list + [labels],
        #     # params=tparams.values(),
        #     # patience=0,
        #     max_gradient_clip=1,
        #     # max_gradient_norm=1,
        #     learning_rate=learning_rate,
        #     monitors=[('cost', cost)],
        #     monitor_gradients=False)

        n_examples = len(train_examples)
        batches_per_epoch = n_examples // options['batch_size'] + 1
#         n_epochs = global_steps // batches_per_epoch + 1
        n_epochs = 10
        global_step = 0
        cost_history = []
        for _ in range(n_epochs):
            for _ in range(batches_per_epoch):
                cost = f_update(*train_loader())
                cost_history += [cost]

                if global_step % disp_freq == 0:
                    print 'global step %d, cost: %f' % (global_step, cost)
                    sys.stdout.flush()
                # dump model parameters.
                if global_step % save_freq == 0:
                    params = unzip(tparams)
                    np.savez(saveto, **params)
                    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)

                # evaluate on test data.
                if global_step % test_freq == 0:
                    scores = evaluate(model['f_prob'], test_loader)
                    print 'eval scores: '
                    print scores
                    end_time = timeit.default_timer()
                    print 'time used: %d seconds.' % (end_time - start_time)
                    sys.stdout.flush()
                global_step += 1

    scores = evaluate(model['f_prob'], test_loader, k_list=[1,5,10,20,50,100],test_batch=True)
    scores["batch_size"] = batch_size
    scores["global_steps"] = global_steps
    scores["disp_freq"] = disp_freq
    scores["save_freq"] = save_freq
    scores["test_freq"] = test_freq
    scores["n_epochs"] = n_epochs
    print scores
    
    sys.stdout.flush()
    with open("test_data_scores_h_nh.json","w") as f:
        json.dump(scores,f,indent=True)


if __name__ == '__main__':
    train(data_dir='data/twitter', dim_proj=512, keep_ratio=1.)