# A comparison of CUDA-accelerated SVM, random forest, and logistic regression binary classification performance

## Abstract
Caruana and Niculescu-Mizil [2006] conducted an empirical comparison of supervised machine learning algorithms as a follow up to a previous study, STATLOG [King et al., 2000]. In the time since this comparison, implementations of these algorithms have greatly changed. Additionally, the development of CUDA-acceleration has made quick training of large data sets feasible. This paper attempts to replicate the results of Caruana and Niculescu-Mizil [2006] by running a subset of algorithms (SVMs, random forests, and logistic regression) on new, modern data sets. The scores of each algorithm over multiple metrics and data sets are presented. This paper also explores the broader implications of GPU machine learning and attempts to quantify training speed over traditional CPU machine learning

## Link to full paper: [pdf](docs/svm_rf_log_reg_CUDA_comp.pdf)