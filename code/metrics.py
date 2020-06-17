'''
Evaluation metrics functions.
'''
# import math
import numpy as np
import collections

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata
from scipy import stats


def _retype(y_prob, y):
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    y_prob = np.array(y_prob)
    y = np.array(y)

    return y_prob, y

# def _retype_hate(y_hate):
#     if not isinstance(y_hate, (collections.Sequence, np.ndarray)):
#         y_hate = [y_hate]
#     y_hate = np.array(y_hate)
#     return y_hate
    
def _binarize(y, n_classes=None):
    return label_binarize(y, classes=range(n_classes))


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(y_prob, y, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    predicted = [np.argsort(p_)[-k:][::-1] for p_ in y_prob]
    actual = [[y_] for y_ in y]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]

    return sum(ranks) / float(len(ranks))


def hits_k(y_prob, y, k=10):
    acc = []
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)

def _apk(actual, pred,k):
    predicted = np.argsort(pred)[-k:][::-1]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def _hits(actual, predicted, k=20):
    predicted = np.argsort(predicted)[-k:][::-1]
    aucc = 0
    for i in predicted:
        if i in actual:
            aucc+=1
    return aucc/ min(len(actual), k)


# def roc_auc(y_prob, y):
#     y = _binarize(y, n_classes=y_prob.shape[1])
#     fpr, tpr, _ = roc_curve(y.ravel(), y_prob.ravel())
#     return auc(fpr, tpr)

# def log_prob(y_prob, y):
#     scores = []
#     for p_, y_ in zip(y_prob, y):
#         assert abs(np.sum(p_) - 1) < 1e-8
#         scores += [-math.log(p_[y_]) + 1e-8]
#         print p_, y_

#     return sum(scores) / len(scores)
def _flatten_y(y_ori, y_len):
    y_flat = []
    for i in range(y_len):
        if i==y_ori:
            y_flat.append(1)
        else:
            y_flat.append(0)
    y_flat = np.array(y_flat)
    return y_flat

def portfolio(y_prob, y, y_hate=None, k_list=[10, 50, 100], test_batch=False):
    y_prob, y = _retype(y_prob, y)
    # scores = {'auc': roc_auc(y_prob, y)}
    # scores = {'mean-rank:': mean_rank(y_prob, y)}
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = hits_k(y_prob, y, k=k)
        scores['map@' + str(k)] = mapk(y_prob, y, k=k)
    if test_batch:
        num_test, y_len = y_prob.shape
        print(num_test, y_len)
        tau_h = 0.0
        row_h = 0.0
        hit_h = 0.0
        map_h = 0.0
        tau_nh = 0.0
        row_nh = 0.0
        hit_nh = 0.0
        map_nh = 0.0
        c_h = 0
        c_nh = 0
        tau =0.0
        row = 0.0
        for i in range(num_test):
            y_flat = _flatten_y(y[i],y_len)
            tau += stats.kendalltau(y_prob[i],y_flat)[0]
            row += stats.spearmanr(y_prob[i],y_flat)[0]
            if y_hate[i]:
                tau_h += stats.kendalltau(y_prob[i],y_flat)[0]
                row_h += stats.spearmanr(y_prob[i],y_flat)[0]
                map_h += _apk(set([y[i]]),y_prob[i],k=20)
                hit_h += _hits(set([y[i]]),y_prob[i],k=20)
                c_h += 1
            else:
                tau_nh += stats.kendalltau(y_prob[i],y_flat)[0]
                row_nh += stats.spearmanr(y_prob[i],y_flat)[0]
                map_nh += _apk(set([y[i]]),y_prob[i],k=20)
                hit_nh += _hits(set([y[i]]),y_prob[i],k=20)
                c_nh += 1
                
        assert c_h+c_nh==num_test
        scores['tau']=tau/num_test
        scores['row']=row/num_test
        scores['hate_tau'] = tau_h/c_h
        scores['hate_row'] = row_h/c_h
        scores['HATE_hits@20'] = hit_h/c_h
        scores['HATE_map@20'] = map_h/c_h
        scores['non_hate_tau'] = tau_nh/c_nh
        scores['non_hate_row'] = row_nh/c_nh
        scores['NON_HATE_hits@20'] = hit_nh/c_nh
        scores['NON_HATE_map@20'] = map_nh/c_nh
    return scores