"""
==========================================================================
Classification of MOABB data

Features: CSP
Classifier: shLDA
==========================================================================
"""

# import packages
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery

root_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches'
df_path = root_path + '/Dataframes/'
fig_path = root_path + '/Figures/'

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

dataset = BNCI2014001()
dataset.subject_list = list(range(1, 10))

# choose paradigm
paradigm = LeftRightImagery()
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

# create pipeline
pipeline = {}
pipeline["LDA"] = make_pipeline(CSP(n_components=8), LDA(solver="lsqr", shrinkage="auto"))

# evaluation
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=dataset, overwrite=False, suffix="newdataset"
)
scores = evaluation.process(pipeline)
print(scores)

results = evaluation.process(pipeline)
results.to_csv(df_path+"CSP_LDA.csv")

fig, ax = plt.subplots(figsize=(8, 7))
results["subj"] = results["subject"].apply(str)
sns.barplot(
    x="score", y="subj", hue="session", data=results, orient="h", palette="viridis", ax=ax
)
plt.show()



