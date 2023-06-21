"""
==========================================================================
Functional connectivity classification

Features: Functional connectivity
Classifiers: LDA and SVC

+ statistical analysis using MOABB
+ concatination of dataframes
+ vizualisation of functional connectivity matrices
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
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, cohen_kappa_score
from matplotlib import pyplot as plt

import moabb
from moabb.paradigms import MotorImagery
from moabb.datasets.base import BaseDataset
import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import compute_dataset_statistics, find_significant_differences

from Scripts.fc_class import FunctionalTransformer, EnsureSPD, GetDataMemory

root_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches'
df_path = root_path + '/Dataframes/'
fig_path = root_path + '/Figures/'

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

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
        data = mat73.loadmat(file_path_list[0]) # Loads the raw data from a .mat file using the loadmat function from the mat73 package.
        x = data["Data_concat_moabb"][subject][0] # Extracts the EEG data for the subject from the loaded data. This data is stored as an array under the key "Data_concat_moabb" in the .mat file. We index into this array using the subject parameter to get the EEG data for that subject. Since the EEG data is stored as a list within the array, we need to use [0] to get the first (and only) element of the list.
        fs = data["fs"] # Extracts the sampling frequency (in Hz) from the loaded data.
        ch_names = data['labels_DK'] # Extracts the channel names for the EEG data
        #subj_ID = data['subject_IDs'][subject]
        events = data["Events_moabb"][subject] # Extracts the event data for the subject from the loaded data

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

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from one subject"""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        file_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/MEG_DK.mat'
        return [file_path]

dataset = MEGdataset()

## list of variables
subject_select = dataset.subject_list[:20] # select the number of subject you want to analyse
spectral_met = ["plv"] # list of spectral metrics
freqbands = {"defaultBand": [8, 35]}
events = ["right_hand", "rest"]

## precompute all metrics for datasets, fc matrix
# Check if the data_fc_all_subjects file exists
if os.path.isfile(df_path+"data_fc_all_subjects"):
    # Load the saved data_fc dictionary from disk
    with open(df_path+"data_fc_all_subjects", "rb") as f:
        data_fc = pickle.load(f)
else:
    data_fc = {} # empty dictionary to store all the functional connectivity matrices
    for f in freqbands: # for each frequency specified in freqbands the code iterates over each subject
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]
        data_fc[f] = {}
        for subject in tqdm(subject_select, desc="subject"):
            data_fc[f][subject] = {} # Creating an empty dictionary as a value for the subject key in the data_fc[f] dictionary
            paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax)
            ep_, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True) # calling get_data that retrieves the preprocessed data for the subject in the form of an Epochs object (ep_), and two arrays (y and metadata)
            for sm in tqdm(spectral_met, desc="met"): # iterating through each element sm in the spectral_met list.
                ft = FunctionalTransformer(delta=1, ratio=0.5, method=sm, fmin=fmin, fmax=fmax) #For each sm (in this case plv), a FunctionalTransformer object is created with specified parameters
                preproc = Pipeline(steps=[("ft", ft), ("spd", EnsureSPD())]) #creating a pipeline object with two steps, ft from above and spd that ensure that the resulting matrices are symmetric positive definite.
                data_fc[f][subject][sm] = preproc.fit_transform(ep_) #The fit_transform method of the preproc pipeline object is applied to the ep_ data, this compute the functional connectivity matrix for the given sm. The resulting functional connectivity matrix is stored in the data_fc dictionary under the f, subject, and sm keys, respectively.
    with open(df_path+"data_fc_all_subjects", "wb") as f:
        pickle.dump(data_fc, f)

## compute results
dataset_res = list()  # creating an empty list to store the results for each subject and frequency
for f in freqbands: # the code iterates over each frequency band
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    for subject in tqdm(subject_select, desc="subject"): # the code iterates over each subject in the list subject_select
        paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax) #A MotorImagery object is created with the current minimum and maximum frequencies.
        ep_, _, _ = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True) # The get_data method is called to extract the motor imagery epochs for the current subject and frequency band from the dataset. The motor imagery epochs, the target labels, and some metadata are returned by the get_data method, but only the epochs are stored in a variable named ep_.
        for sm in tqdm(spectral_met, desc="met"):  # iterating through each element sm in the spectral_met list.
            ft = FunctionalTransformer(delta=1, ratio=0.5, method=sm, fmin=fmin, fmax=fmax)  # For each sm (in this case plv), a FunctionalTransformer object is created with specified parameters

        pipeline = {} #creates an empty dictionary to store the classifier pipelines
        gd = GetDataMemory(subject, f, sm, data_fc) # creates a GetDataMemory object (gd) with the specified subject, frequency range (f), spectral metric (plv), and the functional connectivity matrices previously computed (data_fc).
        pipeline[sm+"-LDA"] = Pipeline(steps=[('gd', gd), ('lda', LDA(solver="lsqr", shrinkage="auto"))])
        pipeline[sm+"-SVC"] = Pipeline(steps=[('gd', gd), ('grid_search_cv', GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=3))])

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
                    }
                    res = {
                        "score": auc,
                        "kappa": kapp,
                        "accuracy": acc,
                        "pipeline": ppn,
                        **res_info,
                    }
                    dataset_res.append(res)
dataset_res = pd.DataFrame(dataset_res)
dataset_res.to_csv(df_path+"fc_LDA_SVC.csv")

## concat the dataframe with the results from classification_comparison
dataset_res = pd.read_csv(df_path+"fc_LDA_SVC.csv")
dataset_CSP = pd.read_csv(df_path+"CSP_LDA_SVC_LR.csv")
result_concat = pd.concat((dataset_res, dataset_CSP))

## statistical analysis
stats = compute_dataset_statistics(result_concat) # based on Wilcoxon tests, takes the dataframe as input and returns a table of p values for each pipeline
print(stats)

# plotting
fig = moabb_plt.meta_analysis_plot(stats, sm+"-LDA", sm+"-SVC")
plt.show()

# comparing different classifiers (makes more sense if I have more pipelines with different classifiers)
P, T = find_significant_differences(stats)
moabb_plt.summary_plot(P, T)
plt.show()

## Averaging accuracy scores across subjects and pipelines
dataset_average = (result_concat.groupby(["subject", "pipeline"])[["accuracy", "score", "kappa"]].agg(["mean", "std"]).reset_index())
print(dataset_average)

## plotting mean functional connectivity matrix for single subject accoss trials
fc_matrix_subject = data_fc['defaultBand'][0]['plv']
mean_matrix = np.mean(fc_matrix_subject, 0)
plt.imshow(mean_matrix, cmap='coolwarm')
plt.title(f'FC Matrix ({f}Hz, {subject}, {sm})')
plt.colorbar()
plt.show()

## plotting functional connectivity matrix for one specific subject, frequency and trial
fc_matrix_single_trial = data_fc['defaultBand'][0]['plv'][0]
plt.imshow(fc_matrix_single_trial, cmap='coolwarm')
plt.title(f'FC Matrix ({f}Hz, {subject}, {sm})')
plt.colorbar()
plt.show()