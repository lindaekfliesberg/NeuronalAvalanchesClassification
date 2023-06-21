"""
===================================================================
Visualization of dataset - Mean avalanches matrix and T-test matrix

+ Group and individual level
+ correction for multiple comparison via FDR
===================================================================
"""

## import packages
import warnings
import os
import mat73
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.io import savemat
from mne.stats import bonferroni_correction, fdr_correction

# Load avalanches to plot avalanche matrices
#data_avalanches = mat73.loadmat('/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/ATM_MEG_DK.mat')
data_avalanches = mat73.loadmat('/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches/Datasets/Opt_ATM_MEG_DK.mat')

root_path = '/Users/linda.ekfliesberg/Documents/GitHub/NeuronalAvalanches'
df_path = root_path + '/Dataframes/'
fig_path = root_path + '/Figures/'

##Plot difference between conditions (rest, right) across trials per subject
# Extract the data and labels
data = data_avalanches["Data_moabb"][0][0]
labels = data_avalanches["Labels_moabb"][0][0]
labels_array = np.array(labels)

# Select the data subsets that correspond to each condition using boolean indexing
rest_data = data[labels_array == 1]
right_data = data[labels_array == 2]

# Compute the means of the data subsets along the 0th axis
rest_mean = np.mean(rest_data, 0)
right_mean = np.mean(right_data, 0)

# Compute the difference matrix
diff_matrix = right_mean - rest_mean

plt.close('all')
# Plot the difference matrix
plt.imshow(diff_matrix, cmap='coolwarm', vmin=-0.2, vmax=0.2)
plt.title('Mean matrix (Right - Rest) for subject_0')
plt.colorbar()
plt.savefig(fig_path+"mean_matrix_subject_0.png", dpi=300)
plt.show()

## Plot difference between conditions (rest, right) across subjects
# Extract the data and labels for all subjects
data = data_avalanches["Data_moabb"]
labels = data_avalanches["Labels_moabb"]

# Compute the mean difference matrix for each subject
diff_matrices = []
for subject in range(len(data)):
    subject_data = data[subject][0]
    subject_labels = labels[subject][0]
    labels_array = np.array(subject_labels)
    rest_data = subject_data[labels_array == 1]
    right_data = subject_data[labels_array == 2]

    rest_mean = np.mean(rest_data, 0)
    right_mean = np.mean(right_data, 0)
    diff_matrix = right_mean - rest_mean
    diff_matrices.append(diff_matrix)

# Compute the mean difference matrix across all subjects
mean_diff_matrix = np.mean(diff_matrices, 0)

plt.close('all')
# Plot the mean difference matrix
plt.imshow(mean_diff_matrix, cmap='coolwarm', vmin=-0.04, vmax=0.04)
plt.title('Mean matrix (Right - Rest) for all subjects')
plt.colorbar()
plt.savefig(fig_path+"mean_matrix_all_subjects.png", dpi=300)
plt.show()

## Plotting matrix containing t-test values for single subject
# Extract the data and labels
data = data_avalanches["Data_moabb"][0][0]
labels = data_avalanches["Labels_moabb"][0][0]
labels_array = np.array(labels)

# Select the data subsets that correspond to each condition using boolean indexing
rest_data = data[labels_array == 1]
right_data = data[labels_array == 2]

# Create an empty matrix to hold the p-values
ttest_matrix = np.zeros((68, 68))
pvalue_matrix = np.zeros((68, 68))

# Loop over all regions of interest
for i in range(68):
    for j in range(68):
        # Extract the data for the current region of interest
        roi_data_rest = rest_data[:, i, j]
        roi_data_right = right_data[:len(roi_data_rest), i, j]

        # Compute the two-sample t-test and extract the t and p
        t_statistic, p_value = stats.ttest_rel(roi_data_right, roi_data_rest)

        # Store the t value in the t-test matrix
        ttest_matrix[i, j] = t_statistic
        pvalue_matrix[i, j] = p_value

# Reshape the p-value matrix to a 1D array
p_values = pvalue_matrix.ravel()
# Perform FDR correction on the p-values
_, pval_bonferroni = bonferroni_correction(p_values, alpha=0.05)
pvalues_bonferroni_matrix = pval_bonferroni.reshape(pvalue_matrix.shape)
_, pval_fdr = fdr_correction(p_values, alpha=0.05, method="indep")
pvalues_fdr_corrected_matrix = pval_fdr.reshape(pvalue_matrix.shape)

# Create a mask to filter p-values less than 0.05
pvalue_mask = pvalue_matrix < 0.05

# Apply the mask to the matrices
filtered_ttest_matrix = np.where(pvalue_mask, ttest_matrix, np.nan)
filtered_pvalue_matrix = np.where(pvalue_mask, pvalue_matrix, np.nan)

# Save the filtered t-test matrix and p-value matrix to MATLAB files
savemat('av_opt_ttest_matrix_subject_2_pvalue<0.05.mat', {'ttest_matrix': filtered_ttest_matrix})
savemat('av_opt_pvalue_matrix_subject_2_pvalue<0.05.mat', {'pvalue_matrix': filtered_pvalue_matrix})

plt.close('all')
# Plot the t-test matrix
plt.imshow(ttest_matrix, cmap='coolwarm', vmin=-5, vmax=5)
plt.title('T-test matrix for single subject')
plt.colorbar()
plt.savefig(fig_path+"ttest_matrix_subject_0.png", dpi=300)
plt.show()

plt.close('all')
# Plot the p-value matrix
plt.imshow(pvalue_matrix, cmap='hot', vmin=0, vmax=0.05)
plt.title('P-value matrix for single subject')
plt.colorbar()
plt.savefig(fig_path+"pvalue_matrix_subject_0.png", dpi=300)
plt.show()

################################################################################
## Plotting matrix containing t-test values for single subject (make it into an array of sums)
# Extract the data and labels
data = data_avalanches["Data_moabb"][0][0]
labels = data_avalanches["Labels_moabb"][0][0]
labels_array = np.array(labels)

# Select the data subsets that correspond to each condition using boolean indexing
rest_data = data[labels_array == 1]
rest_sums = np.sum(rest_data, axis=1)
right_data = data[labels_array == 2]
right_sums = np.sum(right_data, axis=1)

# Create an empty matrix to hold the p-values
ttest_array = np.zeros((68))
pvalue_array = np.zeros((68))

# Loop over all regions of interest
for i in range(68):
    # Extract the data for the current region of interest
    roi_data_rest = rest_sums[:, i]
    roi_data_right = right_sums[:len(roi_data_rest), i]

    # Compute the two-sample t-test and extract the t and p
    t_statistic, p_value = stats.ttest_rel(roi_data_right, roi_data_rest)

    # Store the t value in the t-test matrix
    ttest_array[i] = t_statistic
    pvalue_array[i] = p_value

savemat('ttest_nodes_subject_0.mat', {'ttest_subject_0': ttest_array})
savemat('pvalue_nodes_subject_0.mat', {'pvalue_subject_0': pvalue_array})

################################################################################
## Plotting matrix containing t-test values across all subjects
# Extract the data and labels for all subjects
data = data_avalanches["Data_moabb"]
labels = data_avalanches["Labels_moabb"]

averaged_rest_data_subject = []
averaged_right_data_subject = []
# Compute the t-test and p-value matrices for each subject and store them in the lists
for subject in range(len(data)):
    subject_data = data[subject][0]
    subject_labels = labels[subject][0]
    labels_array = np.array(subject_labels)
    rest_data = subject_data[labels_array == 1]
    right_data = subject_data[labels_array == 2]

    averaged_rest_data = np.mean(rest_data, axis=0)
    averaged_right_data = np.mean(right_data, axis=0)

    averaged_rest_data_subject.append(averaged_rest_data)
    averaged_right_data_subject.append(averaged_right_data)

ttest_matrix = np.zeros((68, 68))
pvalue_matrix = np.zeros((68, 68))

for i in range(68):
    for j in range(68):
        # Extract the data for the current region of interest
        roi_data_rest = [subj[i, j] for subj in averaged_rest_data_subject]
        roi_data_right = [subj[i, j] for subj in averaged_right_data_subject]

        # Compute the two-sample t-test and extract the t and p
        t_statistic, p_value = stats.ttest_rel(roi_data_right, roi_data_rest)

        # Store the t value in the t-test matrix
        ttest_matrix[i, j] = t_statistic
        pvalue_matrix[i, j] = p_value

# Reshape the p-value matrix to a 1D array
p_values = pvalue_matrix.ravel()
# Perform FDR correction on the p-values
_, pval_bonferroni = bonferroni_correction(p_values, alpha=0.05)
pvalues_bonferroni_matrix = pval_bonferroni.reshape(pvalue_matrix.shape)
_, pval_fdr = fdr_correction(p_values, alpha=0.05, method="indep")
pvalues_fdr_corrected_matrix = pval_fdr.reshape(pvalue_matrix.shape)

# Create a mask to filter p-values less than 0.05
pvalue_mask = pvalue_matrix < 0.05

# Apply the mask to the matrices
filtered_ttest_matrix = np.where(pvalue_mask, ttest_matrix, np.nan)
filtered_pvalue_matrix = np.where(pvalue_mask, pvalue_matrix, np.nan)

# Save the filtered t-test matrix and p-value matrix to MATLAB files
savemat('av_opt_ttest_matrix_all_subjects_pvalue<0.05.mat', {'ttest_matrix': filtered_ttest_matrix})
savemat('av_opt_pvalue_matrix_all_subjects_pvalue<0.05.mat', {'pvalue_matrix': filtered_pvalue_matrix})

## Visualization
plt.close('all')
# Plot the t-test matrix
plt.imshow(ttest_matrix, cmap='coolwarm')
plt.title('T-test matrix for all subjects')
plt.colorbar()
plt.savefig(fig_path+"ttest_matrix_all_subjects.png", dpi=300)
plt.show()

plt.close('all')
# Plot the p-value matrix
plt.imshow(pvalue_matrix, cmap='hot', vmin=0, vmax=0.05)
plt.title('P-value matrix for all subjects')
plt.colorbar()
plt.savefig(fig_path+"pvalue_matrix_all_subjects.png", dpi=300)
plt.show()

#######################################################################
## Plotting matrix containing t-test values across all subjects (make it into an array of sums)
# Extract the data and labels for all subjects
data = data_avalanches["Data_moabb"]
labels = data_avalanches["Labels_moabb"]

averaged_rest_data_subject = []
averaged_right_data_subject = []
# Compute the t-test and p-value matrices for each subject and store them in the lists
for subject in range(len(data)):
    subject_data = data[subject][0]
    subject_labels = labels[subject][0]
    labels_array = np.array(subject_labels)
    rest_data = subject_data[labels_array == 1]
    right_data = subject_data[labels_array == 2]

    averaged_rest_data = np.mean(rest_data, axis=0)
    averaged_rest_sums = np.sum(averaged_rest_data, axis=0)
    averaged_right_data = np.mean(right_data, axis=0)
    averaged_right_sums = np.sum(averaged_right_data, axis=0)

    averaged_rest_data_subject.append(averaged_rest_sums)
    averaged_right_data_subject.append(averaged_right_sums)

ttest_array = np.zeros((68))
pvalue_array = np.zeros((68))

for i in range(68):
    # Extract the data for the current region of interest
    roi_data_rest = [subj[i] for subj in averaged_rest_data_subject]
    roi_data_right = [subj[i] for subj in averaged_right_data_subject]

    # Compute the two-sample t-test and extract the t and p
    t_statistic, p_value = stats.ttest_rel(roi_data_right, roi_data_rest)

    # Store the t value in the t-test matrix
    ttest_array[i] = t_statistic
    pvalue_array[i] = p_value

savemat('ttest_nodes_all_subjects.mat', {'ttest_all_subjects': ttest_array})
savemat('pvalue_nodes_all_subjects.mat', {'pvalue_all_subjects': pvalue_array})

