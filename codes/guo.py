import numpy as np
import scipy.stats


def mean_signal(t, m, e):
    return np.mean(m)


def std_signal(t, m, e):
    return np.std(m)


def mean_square_signal(t, m, e):
    return np.mean(m ** 2)


def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))


def skew_signal(t, m, e):
    return scipy.stats.skew(m)


guo_features = {
    "mean": mean_signal,
    "std": std_signal,
    "mean2": mean_square_signal,
    "abs_diffs": abs_diffs_signal,
    "skew": skew_signal,
}
