import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import levinson_durbin_pacf as ar_from_pacf

from timeseries.analysis import acf, pacf


# pacf methods definitions and descriptions
ols = "ols"
burg = "burg"
ld_biased = "ld_biased"

methods_descrs = {
    ols: "Regression of time series on lags of it and on constant",
    burg: "Burg's partial autocorrelation estimator",
    ld_biased: "Levinson-Durbin recursion without bias correction",
}


def ar_coeffs(x, p, method):
    pacf_values = pacf(x, method=method, nlags=p)
    ar_coeffs, _ = ar_from_pacf(pacf_values, nlags=p)
    return ar_coeffs


def ar_one_step_pred(x, y, ar_coeffs, rand_std=None):
    """
    If rand_std is not None, then a random error of N(0,rand_std^2) is added to each predicted value.
    """
    n = len(x)
    ar_flipped = np.flipud(ar_coeffs)
    p = len(ar_flipped)
    y[:p] = x[:p]
    for i in range(p, n):
        d = np.dot(x[i - p : i], ar_flipped)
        if rand_std is not None:
            d += float(np.random.normal(0, rand_std, 1))
        y[i] = d


def psd_fun_from_ar(ar_coeffs, ts, sampT):
    p = len(ar_coeffs)
    if type(ts) is pd.Series:
        x = ts.values
    else:
        x = ts

    y = np.zeros_like(x)
    ar_one_step_pred(x, y, ar_coeffs)
    s = np.std(x - y) ** 2

    return np.vectorize(
        lambda f: s
        * sampT
        / (
            np.abs(
                (
                    1
                    - np.dot(
                        np.exp(-2j * np.pi * f * sampT * np.arange(1, p + 1)),
                        ar_coeffs,
                    )
                )
            )
            ** 2
        )
    )


def ar_features_fun(p, method, k):
    return lambda t, m, e: ar_coeffs(m, p, method)[k]


def psd_features_fun(freq, N, k, method, p):
    f = k / (2 * N) * freq
    return lambda t, m, e: psd_fun_from_ar(ar_coeffs(m, p, method), m, 1.0 / freq)(f)


def empty_fset(ts, cols):
    return pd.DataFrame(
        [],
        index=np.arange(len(ts[0])),
        columns=pd.MultiIndex.from_tuples(cols, names=["feature", "channel"]),
    )


def ts_and_ftrs_channels_cols(ts, ftr_name_fun, ftrs, fset=None):
    if type(ts[0]) is np.ndarray:
        ts = [[x] for x in ts]
    channels = len(ts[0])
    cols = [(ftr_name_fun(k), ch) for ch in range(channels) for k in range(ftrs)]
    new_fset = empty_fset(ts, cols)
    if fset is not None:
        new_fset = pd.concat([fset, new_fset])
    return ts, new_fset


def ar_features(ts, p, method, fset=None):
    def ftr_name(k):
        return f"AR({p})_{k+1}"

    ts, fset = ts_and_ftrs_channels_cols(ts, ftr_name, p, fset)
    #     print(f"ts:\n{ts}\n")
    for i, xs in enumerate(ts):
        #         print(f"xs:\n{xs}\n")
        for ch, x in enumerate(xs):
            #             print(f"x:\n{x}\n")
            coeffs = ar_coeffs(x, p, method)
            for k in range(p):
                fset.loc[i, (ftr_name(k), ch)] = coeffs[k]
    return fset


def psd_ar_features(ts, freq, N, p, method, fset=None):
    def ftr_name(k):
        return f"psd_{k}_AR({p})"

    ts, fset = ts_and_ftrs_channels_cols(ts, ftr_name, N, fset)

    f_coeff = 1 / (2 * N) * freq
    for i, xs in enumerate(ts):
        for ch, x in enumerate(xs):
            coeffs = ar_coeffs(x, p, method)
            psd_fun = psd_fun_from_ar(coeffs, x, 1 / freq)
            for k in range(N):
                fset.loc[i, (ftr_name(k), ch)] = psd_fun(k * f_coeff)
    return fset
