"""
==========================================================================
Classification of MEG data

Features: CSP
Classifier: shLDA
==========================================================================
"""

# import packages
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import mat73
import mne
from tqdm import tqdm

import moabb
from moabb.datasets.base import BaseDataset
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# create class for the MEG data
class MEGdataset(BaseDataset):

    def __init__(self):
        super().__init__(
            subjects=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
            sessions_per_subject=1,
            events={"right_hand": 1, "rest": 2},
            code="MEG dataset",
            interval=[3, 6],
            paradigm="imagery",
            doi="",
        )

    def _get_single_subject_data(self, subject):

        file_path_list = self.data_path(subject)
        data = mat73.loadmat(file_path_list[0])
        x = data["Data_concat_moabb"][subject][0]
        fs = data["fs"]
        ch_names = data['labels_DK']
        #subj_ID = data['subject_IDs'][subject]
        events = data["Events_moabb"][subject]

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(data=np.array(x), info=info)

        mapping = {1: 'right_hand', 2: 'rest'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)

        #sessions = {}
        #sessions["session_1"] = {}
        #sessions["session_1"]["run_1"] = raw
        return {"session_0": {"run_0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from one subject"""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        file_path = "/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/MEG_DK.mat"
        return [file_path]

dataset = MEGdataset()
dataset.subject_list = dataset.subject_list[:20]
subject = dataset.subject_list

# choose paradigm
paradigm = MotorImagery()
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

# create pipeline
pipeline = {}
pipeline["CSP"+"-LDA"] = make_pipeline(CSP(n_components=8), LDA(solver="lsqr", shrinkage="auto"))

# # evaluation
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=dataset, overwrite=True, suffix="newdataset"
)

results = evaluation.process(pipeline)
# results.to_csv("./CSP_LDA.csv")
# results = pd.read_csv("./CSP_LDA.csv")

fig, ax = plt.subplots(figsize=(8, 7))
results["subj"] = results["subject"].apply(str)
sns.barplot(
    x="score", y="subj", hue="session", data=results, orient="h", palette="viridis", ax=ax
)
plt.show()
