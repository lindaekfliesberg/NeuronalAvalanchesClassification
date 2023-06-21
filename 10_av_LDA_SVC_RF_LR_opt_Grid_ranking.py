"""
==========================================================================
Neuronal avalanches - Random forest

+ adding random forest attribute feature importance ranking in the loop
==========================================================================
"""

## import packages
import warnings

import mne
import mat73
import pickle
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from matplotlib import pyplot as plt
from scipy.io import savemat

import moabb
from moabb.paradigms import MotorImagery
from moabb.datasets.base import BaseDataset
import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import compute_dataset_statistics, find_significant_differences

from Scripts.fc_class import FunctionalTransformer, EnsureSPD, GetDataMemory, GetAvalanches, GetAvalanchesNodal

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

root_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches'
df_path = root_path + '/Dataframes/'
fig_path = root_path + '/Figures/'

## create class for the MEG data
class MEGdataset(BaseDataset):

    def __init__(self):
        super().__init__(
            subjects=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
            sessions_per_subject=1,
            events={"right_hand": 1, "rest": 2}, # a dictionary mapping event labels to integers
            code="MEG dataset", # a string representing the dataset code
            interval=[3, 6], # a list representing the interval of time in which the events occur
            paradigm="imagery",
            doi="", # a string representing the DOI (Digital Object Identifier) of the dataset (here left blank)
        )

    def _get_single_subject_data(self, subject): #  takes in a subject parameter and returns a dictionary representing the raw data for that subject.

        file_path_list = self.data_path(subject)
        data_meg = mat73.loadmat(file_path_list[0]) # Loads the raw data from a .mat file using the loadmat function from the mat73 package.
        data_avalanches = mat73.loadmat(file_path_list[1])
        x = data_meg["Data_concat_moabb"][subject][0] # Extracts the EEG data for the subject from the loaded data. This data is stored as an array under the key "Data_concat_moabb" in the .mat file. We index into this array using the subject parameter to get the EEG data for that subject. Since the EEG data is stored as a list within the array, we need to use [0] to get the first (and only) element of the list.
        fs = data_meg["fs"] # Extracts the sampling frequency (in Hz) from the loaded data.
        ch_names = data_meg['labels_DK'] # Extracts the channel names for the EEG data
        #subj_ID = data['subject_IDs'][subject]
        events = data_avalanches["Events_moabb"][subject] # Extracts the event data for the subject from the loaded data

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])] # use a list comprehension to create a list with the same length as the number of channel names. Here eeg is selected as channel type instead of meg but this doesn't have any neurophysiological meaning. It's only to have an convention
        info = mne.create_info(ch_names, fs, ch_types) # Creates an info object using the create_info function from the mne package.
        raw = mne.io.RawArray(data=np.array(x), info=info) # Creates a RawArray object using the RawArray function from the mne package, passing in the EEG data and info object.

        mapping = {1: 'right_hand', 2: 'rest'} # Creates a mapping from event integers to event labels. In this case, events with the integer value 1 are labeled as "right_hand" and events with the integer value 2 are labeled as "rest".
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq']) # Creates a mne.Annotations object from the event data using the annotations_from_events function from the mne package.
        raw.set_annotations(annot_from_events) # Adds the annotations to the RawArray object using the set_annotations method.

        #sessions = {}
        #sessions["session_1"] = {}
        #sessions["session_1"]["run_1"] = raw
        return {"session_0": {"run_0": raw}}

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """Download the data from one subject"""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        meg_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/MEG_DK.mat'
        #avalanches_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/ATM_MEG_DK.mat'
        avalanches_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/Opt_ATM_MEG_DK.mat'
        return [meg_path, avalanches_path]

dataset = MEGdataset()

# Load avalanches to plot avalanche matrices
#data_avalanches = mat73.loadmat('/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/ATM_MEG_DK.mat')
data_avalanches = mat73.loadmat('/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/Opt_ATM_MEG_DK.mat')

# list of variables
subject_select = dataset.subject_list[:20] # select the number of subject you want to analyse
freqbands = {"defaultBand": [8, 35]}
events = ["right_hand", "rest"]

## compute results
dataset_av = list()  # creating an empty list to store the results for each subject and frequency
for f in freqbands: # the code iterates over each frequency band
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    for subject in tqdm(subject_select, desc="subject"): # the code iterates over each subject in the list subject_select
        paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax) #A MotorImagery object is created with the current minimum and maximum frequencies.

        #avalanches_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/ATM_MEG_DK.mat'
        avalanches_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/Opt_ATM_MEG_DK.mat'

        #ga = GetAvalanches(subject, avalanches_path) # creates a GetDataMemory object (gd) with the specified subject, frequency range (f), spectral metric (plv), and the functional connectivity matrices previously computed (data_av).
        gan = GetAvalanchesNodal(subject, avalanches_path)

        pipeline = {} #creates an empty dictionary to store the classifier pipelines
        #pipeline["avn"+"-LDA"] = Pipeline(steps=[('gav', gav), ('lda', LDA(solver="lsqr", shrinkage="auto"))])
        #pipeline["avn"+"-SVC"] = Pipeline(steps=[('gav', gav), ('svc', GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=3))])
        pipeline["avn"+"-RF"] = Pipeline(steps=[('gan', gan), ('rf', GridSearchCV(RandomForestClassifier(), {'n_estimators': [70,75,80,85,90], 'min_samples_leaf': [1,2,3], 'max_features': ['sqrt','log2'], 'max_depth': [3,5,7,9], 'random_state': [42]}, cv=5))])
        #pipeline["avn"+"-LR"] = Pipeline(steps=[('gav', gav), ('lr', LogisticRegression(penalty="elasticnet", l1_ratio=0.15, intercept_scaling=1000.0, solver="saga"))])

        # Train and evaluate
        _, y, metadata = paradigm.get_data(dataset, [subject], return_epochs=True) # _ is the epoches object, y is an array of labels
        X = np.arange(len(y)) #X is created as an array of integers ranging from 0 to the length of y
        for session in np.unique(metadata.session): # the code iterates over each session in the metadata (in this case only one session)
            ix = metadata.session == session # ix is assigned to a boolean array indicating the indices in metadata.session where the value is equal to session.
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            le = LabelEncoder() # le is created to encode the labels
            y_ = le.fit_transform(y[ix]) # le  fitted to the labels of the current session (y[ix])
            X_ = X[ix]

            for idx, (train, test) in enumerate(cv.split(X_, y_)): # The code iterates over each split of the cross-validator (cv.split(X_, y_)).
                for ppn, ppl in tqdm(pipeline.items(), total=len(pipeline), desc="pipelines"): #The code iterates over each pipeline (pipeline.items()), where pipeline is a dictionary of classifier pipelines.
                    cvclf = clone(ppl) #A clone of the current pipeline (clone(ppl)) is created
                    cvclf.fit(X_[train], y_[train]) # The clone is fitted to the training data (cvclf.fit(X_[train], y_[train])).
                    yp = cvclf.predict(X_[test]) #The predictions of the classifier on the test data (cvclf.predict(X_[test])) are assigned to yp.
                    acc = balanced_accuracy_score(y_[test], yp) # The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.
                    auc = roc_auc_score(y_[test], yp) # ROC AUC (Area Under the Receiver Operating Characteristic Curve) is a measure of the trade-off between true positive rate (TPR) and false positive rate (FPR) at different classification thresholds. It is commonly used to evaluate binary classifiers and is a useful metric when the classes are not heavily imbalanced.
                    kapp = cohen_kappa_score(y_[test], yp) # measures the agreement between the observed and the expected agreement between two raters, accounting for the agreement that could be expected by chance.
                    importances = cvclf.named_steps["rf"].best_estimator_.feature_importances_

                    res_info = {
                        "subject": subject,
                        "session": "session_0",
                        "n_sessions": 1,
                        "FreqBand": "defaultBand",
                        "dataset": dataset.code,
                        "fmin": fmin,
                        "fmax": fmax,
                        "samples": len(y_),
                        "time": 0.0,
                        "split": idx,
                        "importances": importances,
                    }
                    res = {
                        "score": auc,
                        "kappa": kapp,
                        "accuracy": acc,
                        "pipeline": ppn,
                        **res_info,
                    }
                    dataset_av.append(res)

dataset_av = pd.DataFrame(dataset_av)
dataset_av.to_csv(df_path+"avn_RF_feature_importance_ranking.csv")

## concat the dataframe with the results from classification_comparison
dataset_CSP = pd.read_csv(df_path+"CSP_LDA.csv")
dataset_fc = pd.read_csv(df_path+"fc_LDA_SVC.csv")
dataset_av = pd.read_csv(df_path+"av_LDA_SVC_RF_LR.csv")
dataset_avn = pd.read_csv(df_path+"avn_LDA_SVC_RF_LR.csv")
dataset_av_opt = pd.read_csv(df_path+"av_opt_LDA_SVC_RF_LR_tuned_hyperparameters.csv")
dataset_avn_opt = pd.read_csv(df_path+"avn_opt_LDA_SVC_RF_LR_tuned_hyperparameters.csv")

concat_fc_av = pd.concat((dataset_av, dataset_fc))
concat_CSP_fc_av_avn = pd.concat((dataset_CSP, dataset_fc, dataset_av, dataset_avn, dataset_av_opt, dataset_avn_opt))

## Feature importance ranking
importances_matrix = np.vstack(dataset_av["importances"])
median_importances = np.median(importances_matrix, axis=0)
labels = data_avalanches['labels_DK'] # Get the labels for the features

#Saving median importances in a matlab file (for mcc to plot on topographical map)
#savemat('avn_opt_median_importance_all_subject.mat', {'median_importances': median_importances})

# Rank from large to small values (used for thresholding)
indices = np.argsort(median_importances)[::-1] # Rank the feature importances and print them
indices = pd.DataFrame(indices)
indices.to_csv(df_path+"avn_feature_indices_median.csv")

# # Rank from small the large values
# indices_reversed = np.argsort(median_importances) # Rank the feature importances and print them
# indices_reversed = pd.DataFrame(indices_reversed)
# indices_reversed.to_csv(df_path+"avn_feature_indices_reversed_median.csv")

# Get the top 20 features and their importance scores
top_20_indices = indices[:20]
top_20_features = [labels[i] for i in top_20_indices]
top_20_scores = median_importances[top_20_indices]

# Plotting
plt.figure(figsize=(8, 6))
plt.barh(top_20_features, top_20_scores, color='#FA8072')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.title('Top 20 Features by Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()