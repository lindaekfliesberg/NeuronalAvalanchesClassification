# NeuronalAvalanchesClassification

This master thesis was based on the following [manuscript](https://www.biorxiv.org/content/10.1101/2022.06.14.495887v1) and associated [code](https://github.com/mccorsi/NeuronalAvalanches4BCI).

## Contacts
- Linda Ek Fliesberg, lindaekfliesberg@gmail.com
- Marie-Constance Corsi, marie.constance.corsi@gmail.com

## Abstract

Brain-computer interface (BCI) technology enables direct communication between the human brain and computers. However, the high inter-subject variability has hindered its efficiency. This study proposes to address the limitations of traditional BCI systems by proposing a novel approach that incorporates neuronal avalanches as alternative features to enhance BCI performance. Neuronal avalanches, synchronized cascades of neural activity propagating across brain regions, offer a comprehensive representation of the brain's complexity. By integrating the temporal and spatial characteristics of neuronal avalanches into the BCI pipeline using MEG, a more nuanced understanding of cognitive states and intentions can be achieved. In this study, we demonstrate the significance of feature selection in BCI pipelines. Our results reveal that the choice of features, including parameters defining neuronal avalanches (e.g., minimum avalanche duration and z-score threshold) and relevant feature selection techniques, substantially impact BCI performance more than the choice of classification methods. Moreover, this research identifies key regions of interest, such as the premotor, motor, and parietal areas, during motor imagery tasks, shedding light on the underlying mechanisms of BCI performance. By utilizing neuronal avalanches as alternative features, our study provides insights into improving BCI classification performance and potentially enhance our understanding of BCI functionality.

## Description of code

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
