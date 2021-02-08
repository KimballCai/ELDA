# EDUTEM: Learning explicit dual-interaction for healthcareanalytics

The source code for ___EDUTEM: Learning explicit dual-interaction for healthcareanalytics___

## Requirements

Install python, tensorflow 1.14.0, Keras 2.3.1.
Detailed requirements are shown in the [requirements.txt](./requirements.txt)


## Data preparation

As for the Physionet2012 dataset, it is a public dataset and you can download it from [https://physionet.org/content/challenge-2012/1.0.0/](https://physionet.org/content/challenge-2012/1.0.0/)
and then preprocess the dataset with [preprocess_phy2012.ipynb](dataset/physionet2012/preprocess_phy2012.ipynb).

As fpr the MIMIC3 dataset, you must submit the application for data access from [https://mimic.physionet.org/](https://mimic.physionet.org/).
After downloading the CSVs, you first need to build benchmark dataset according to the [https://github.com/YerevaNN/mimic3-benchmarks/](https://github.com/YerevaNN/mimic3-benchmarks/),
and then please preprocess the dataset with [preprocess_MIMIC3.ipynb](dataset/MIMIC3/preprocess_MIMIC3.ipynb).


## Training EDUTEM

You can run the EDUTEM with the jupyter file [EDUTEM_Train.ipynb](EDUTEM_Train.ipynb), and we have already set up the default setting to train the EDUTEM on the physionet2012 dataset.

1. Please specify the parameter for the dataset setting, such as dataset, application, data filing, standardization.
2. Please specify the parameters for the model setting, such as embed_dim, hidden_dim, clip_min, clip_mac, compress_dim.
3. Please specify the parameters for the training setting, such as batch_size, learning_rate, patience.

If you want to use your own dataset, please design the Dataloader like it in physionet2012 or MIMIC3, and do the forward imputation (i.e. impute the missing data with the last observation) and standardization before training the EDUTEM.
