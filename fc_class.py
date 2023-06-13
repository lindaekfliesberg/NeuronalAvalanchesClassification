"""
===================================================================================================
 Classes used to load the features (functional connectivity and neuronal avalanches) into the pipelines

 - GetDataMemory is used to load the precomputed functional connectivity matrices and flatten them into an array
 - GetAvalanches is used to load the precomputed neuronal avalanches and flatten them into an array
 - GetAvalanchesNodal is used to load the precomputed neuronal avalanches, sum all the element in each row, output as an array
 - GetAvalanchesNodalThreshold is used to load the precomputed neuronal avalanches, sum all the element in each row, output as an array and then threshold the top ranking features
===================================================================================================
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#          Linda Ek Fliesberg <lindaekfliesberg@gmail.com>

from sklearn.covariance import ledoit_wolf
from sklearn.base import BaseEstimator, TransformerMixin

import hashlib
import os.path as osp
import os
import mat73

from mne import get_config, set_config, set_log_level, EpochsArray
from mne.connectivity import spectral_connectivity
from mne.connectivity import envelope_correlation
from moabb.evaluations.base import BaseEvaluation
from scipy import stats as spstats

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from time import time
import numpy as np
from mne.epochs import BaseEpochs
from sklearn.metrics import get_scorer

from pyriemann.classification import FgMDM
from pyriemann.estimation import Coherences


def _compute_fc_subtrial(epoch, delta=1, ratio=0.5, method="coh", fmin=8, fmax=35):
    """Compute single trial functional connectivity (FC)

    Most of the FC estimators are already implemented in mne-python (and used here from
    mne.connectivity.spectral_connectivity and mne.connectivity.envelope_correlation).
    The epoch is split into subtrials.

    Parameters
    ----------
    epoch: MNE epoch
        Epoch to process
    delta: float
        length of the subtrial in seconds
    ratio: float, in [0, 1]
        ratio overlap of the sliding windows
    method: string
        FC method to be applied, currently implemented methods are: "coh", "plv",
        "imcoh", "pli", "pli2_unbiased", "wpli", "wpli2_debiased", "cov", "plm", "aec"
    fmin: real
        filtering frequency, lowpass, in Hz
    fmax: real
        filtering frequency, highpass, in Hz

    Returns
    -------
    connectivity: array, (nb channels x nb channels)

    #TODO: compare matlab/python plm's output
    The only exception is the Phase Linearity Measurement (PLM). In this case, it is a
    Python version of the ft_connectivity_plm MATLAB code [1] of the Fieldtrip
    toolbox [2], which credits [3], with the "translation" into Python made by
    M.-C. Corsi.

    references
    ----------
    .. [1] https://github.com/fieldtrip/fieldtrip/blob/master/connectivity/ft_connectivity_plm.m  # noqa
    .. [2] R. Oostenveld, P. Fries, E. Maris, J.-M. Schoffelen, and  R. Oostenveld,
    "FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive
    Electrophysiological  Data" (2010): https://doi.org/10.1155/2011/156869
    .. [3] F. Baselice, A. Sorriso, R. Rucco, and P. Sorrentino, "Phase Linearity
    Measurement: A Novel Index for Brain Functional Connectivity" (2019):
    https://doi.org/10.1109/TMI.2018.2873423
    """
    lvl = set_log_level("CRITICAL")
    L = epoch.times[-1] - epoch.times[0]
    sliding = ratio * delta
    # fmt: off
    spectral_met = ["coh", "plv", "imcoh", "pli", "pli2_unbiased",
                    "wpli", "wpli2_debiased", ]
    other_met = ["cov", "plm", "aec"]
    # fmt: on
    if not method in spectral_met + other_met:
        raise NotImplemented("this spectral connectivity method is not implemented")

    sfreq, nb_chan = epoch.info["sfreq"], epoch.info["nchan"]
    win = delta * sfreq
    nb_subtrials = int(L * (1 / (sliding + delta) + 1 / delta))
    nbsamples_subtrial = delta * sfreq

    # X, total nb trials over the session(s) x nb channels x nb samples
    X = np.squeeze(epoch.get_data())
    subtrials = np.empty((nb_subtrials, nb_chan, int(win)))

    for i in range(0, nb_subtrials):
        idx_start = int(sfreq * i * sliding)
        idx_stop = int(sfreq * i * sliding + nbsamples_subtrial)
        subtrials[i, :, :] = np.expand_dims(X[:, idx_start:idx_stop], axis=0)
    sub_epoch = EpochsArray(np.squeeze(subtrials), info=epoch.info)
    if method in spectral_met:
        r = spectral_connectivity(
            sub_epoch,
            method=method,
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            tmin=0,
            mt_adaptive=False,
            n_jobs=1,
        )
        c = np.squeeze(r[0])
        c = c + c.T - np.diag(np.diag(c)) + np.identity(nb_chan)
    elif method == "aec":
        # filter in frequency band of interest
        sub_epoch.filter(
            fmin,
            fmax,
            n_jobs=1,
            l_trans_bandwidth=1,  # make sure filter params are the same
            h_trans_bandwidth=1,
        )  # in each band and skip "auto" option.
        # apply hilbert transform first
        h_sub_epoch = sub_epoch.apply_hilbert()
        c = envelope_correlation(h_sub_epoch, verbose=True)
        # by default, combine correlation estimates across epochs by peforming an average
        # output : nb_channels x nb_channels -> no need to rearrange the matrix
    elif method == "cov":
        c = ledoit_wolf(X.T)[0]  # oas ou fast_mcd

    return c


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def isPD2(B):
    """Returns true when input is positive-definite, via eigenvalues"""
    if np.any(np.linalg.eigvals(B) < 0.0):
        return False
    else:
        return True


def nearestPD(A, reg=1e-6):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): htttps://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        # Regularize if already PD
        ei, ev = np.linalg.eigh(A3)
        if np.min(ei) / np.max(ei) < reg:
            A3 = ev @ np.diag(ei + reg) @ ev.T
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD2(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    # Regularize
    ei, ev = np.linalg.eigh(A3)
    if np.min(ei) / np.max(ei) < reg:
        A3 = ev @ np.diag(ei + reg) @ ev.T
    return A3


class FunctionalTransformer(TransformerMixin, BaseEstimator):
    """Getting connectivity features from epoch"""

    def __init__(self, delta=1, ratio=0.5, method="coh", fmin=8, fmax=35):
        self.delta = delta
        self.ratio = ratio
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        if get_config("MOABB_PREPROCESSED") is None:
            set_config(
                "MOABB_PREPROCESSED",
                osp.join(osp.expanduser("~"), "mne_data", "preprocessing"),
            )
        if not osp.isdir(get_config("MOABB_PREPROCESSED")):
            os.makedirs(get_config("MOABB_PREPROCESSED"))
        self.preproc_dir = get_config("MOABB_PREPROCESSED")
        self.cname = "-".join(
            [
                str(e)
                for e in [
                self.method,
                self.delta,
                self.ratio,
                self.fmin,
                self.fmax,
                ".npz",
            ]
            ]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # StackingClassifier uses cross_val_predict, that apply transform
        # with dispatch_one_batch, streaming each trial one by one :'(
        # If training on a whole set, cache results otherwise compute
        # fc each time

        #load data

        if isinstance(X, BaseEpochs):
            if self.method in ['instantaneous', 'lagged']:
                Xfc_temp = Coherences(coh=self.method, fmin=self.fmin, fmax=self.fmax,
                                      fs=X.info["sfreq"]).fit_transform(X.get_data())
                Xfc = np.empty(Xfc_temp.shape[:-1], dtype=Xfc_temp.dtype)
                for trial, fc in enumerate(Xfc_temp):
                    Xfc[trial, :, :] = fc.mean(axis=-1)
                return Xfc

            fcache = hashlib.md5(X.get_data()).hexdigest() + self.cname
            if osp.isfile(fcache):
                return np.load(fcache)["Xfc"]
            else:
                Xfc = np.empty((len(X), X[0].info["nchan"], X[0].info["nchan"]))
                for i in range(len(X)):
                    Xfc[i, :, :] = _compute_fc_subtrial(
                        X[i],
                        delta=self.delta,
                        ratio=self.ratio,
                        method=self.method,
                        fmin=self.fmin,
                        fmax=self.fmax,
                    )

            return Xfc


class EnsureSPD(TransformerMixin, BaseEstimator):
    """Getting connectivity features from mat files"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xspd = np.empty_like(X)
        for i, mat in enumerate(X):
            Xspd[i, :, :] = nearestPD(mat)
        return Xspd

    def fit_transform(self, X, y=None):
        transf = self.transform(X)
        return transf

# GetDataMemory takes the precomputed functional connectivity matrices and flattens them as an array
class GetDataMemory(TransformerMixin, BaseEstimator):
    """Get data for ensemble"""
    def __init__(self, subject, freqband, method, precomp_data):
        self.subject = subject
        self.freqband = freqband
        self.method = method
        self.precomp_data = precomp_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        temp=self.precomp_data[self.freqband][self.subject][self.method][X]
        matrix_fc = temp
        array_fc = np.reshape(matrix_fc, (np.shape(matrix_fc)[0], np.shape(matrix_fc)[1] * np.shape(matrix_fc)[2]))
        return array_fc

    def fit_transform(self, X, y=None):
        return self.transform(X)

# GetAvalanches load the precomputed neuronal avalanches and flatten them into an array
class GetAvalanches(TransformerMixin, BaseEstimator):
    """Get neuronal avalanches and flatten the matrices"""
    def __init__(self, subject, avalanches_path):
        self.subject = subject
        self.avalanches_path = avalanches_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = mat73.loadmat(self.avalanches_path)
        matrix_av = data["Data_moabb"][self.subject][0][X]
        array_av = np.reshape(matrix_av, (np.shape(matrix_av)[0], np.shape(matrix_av)[1] * np.shape(matrix_av)[2]))
        return array_av

    def fit_transform(self, X, y=None):
        return self.transform(X)


# GetAvalanchesNodal load the precomputed neuronal avalanches and sum all the element in each row, output as a vector
class GetAvalanchesNodal(TransformerMixin, BaseEstimator):
    """Get neuronal avalanches and flatten the matrices"""
    def __init__(self, subject, avalanches_path):
        self.subject = subject
        self.avalanches_path = avalanches_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = mat73.loadmat(self.avalanches_path)
        matrix_av = data["Data_moabb"][self.subject][0][X]
        row_sums = np.sum(matrix_av, axis=1)
        return row_sums

    def fit_transform(self, X, y=None):
        return self.transform(X)

# GetAvalanchesVector load the precomputed neuronal avalanches and sum all the element in each row, output as a vector and then threshold the top ranking features
class GetAvalanchesNodalThreshold(TransformerMixin, BaseEstimator):
    """Get neuronal avalanches and flatten the matrices"""
    def __init__(self, subject, avalanches_path, features, ranking):
        self.subject = subject
        self.avalanches_path = avalanches_path
        self.features = features
        self.ranking = ranking

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = mat73.loadmat(self.avalanches_path)
        matrix_av = data["Data_moabb"][self.subject][0][X]
        row_sums = np.sum(matrix_av, axis=1)

        selected_features = self.ranking.iloc[:self.features, 1].values
        row_sums_reduc = np.take(row_sums, selected_features, axis=1)
        return row_sums_reduc

    def fit_transform(self, X, y=None):
        return self.transform(X)
