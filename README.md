# NeuronalAvalanchesClassification
- 1_CSP_LDA_MOABBdata.py contains a toy pipeline using CSP+LDA on data avalible at MOABB.

- 2_CSP_LDA_MEGdata.py contains a toy pipeline using CSP+LDA on MEG data.

- 3_CSP_LDA_SVC_LR.py contains a pipeline using CSP combined with LDA, SVM and LR.

- 4_fc_LDA_SVC_concat.py contains a pipeline for precomputing functional connectivity matrices and applying them to LDA and SVM.

- 5_av_LDA_SVC_RF_LR.py contains a pipeline that takes precomputed neuronal avalanche transition matrices and apply them to LDA, SVM, RF and LR.

- 6_av_RF_n_estimators_opt.py is used to tune random forest hyperparameter n_estimators.

- 7_av_RF_min_samples_leaf_opt.py is used to tune random forest hyperparameter min_samples_leaf.

- 8_av_RF_max_depth_opt.py is used to tune random forest hyperparameter max_depth.

- 9_av_LDA_SVC_RF_LR_opt_Grid.py contains a pipeline that takes precomputed neuronal avalanche transition matrices and apply them to LDA, SVM, RF and LR after tuning of the hyperparameters.

- 10_av_LDA_SVC_RF_LR_opt_Grid_ranking.py uses the random forest pipeline to compute the feature importance ranking that is used for feature selection.

- 11_av_LDA_SVC_RF_LR_opt_Grid_thresholding.py performs features selection by testing the optimal number of features to include in the pipeline.

- 12_ttest_matrix.py takes the precomputed neuronal avalanche transition matrices and compute the t-test between conditions.

- fc_class.py contains all the classes that is used in the above scripts.
