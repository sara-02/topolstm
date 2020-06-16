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

def portfolio(y_prob, y, k_list=[10, 50, 100], test_batch=False):
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
        tau = 0.0
        row= 0.0
        for i in range(num_test):
            y_flat = _flatten_y(y[i],y_len)
            tau += stats.kendalltau(y_prob[i],y_flat)[0]
            row += stats.spearmanr(y_prob[i],y_flat)[0]
        scores['tau'] = tau/num_test
        scores['row'] = row/num_test
    return scores