__author__ = 'ltorres'

import pandas
import numpy as np


#
# Distance functions.  All distance functions must take care to handle missing data.
#


def dist_euclidean(v1, v2):
    """
    Euclidean distance.
    """
    return sum([(x1-x2)**2 for x1, x2 in zip(v1, v2) if not pandas.isnull(x1) and not pandas.isnull(x2)])


def mu_euclidean(k, cluster, d):
    """
    Euclidean mean.
    """
    mu = cluster.mu * 0

    for i in range(d):
        norm_i = 0
        for x in cluster.data:
            if not np.isnan(x[i]):
                mu[i] += x[i]
                norm_i += 1
        mu[i] /= norm_i

    return (k, mu)

