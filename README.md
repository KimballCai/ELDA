# ELDA: Learning explicit dual-interaction for healthcare analytics

The source code for ___ELDA: Learning Explicit Dual-Interactions for Healthcare Analytics___

## Requirements

Install python 3.6, TensorFlow 1.14.0 , Keras 2.3.1.


## Data preparation

As for the Physionet2012 dataset, it is a public dataset and you can download it from [https://physionet.org/content/challenge-2012/1.0.0/](https://physionet.org/content/challenge-2012/1.0.0/)
and then preprocess the dataset with [preprocess_phy2012.ipynb](dataset/physionet2012/preprocess_phy2012.ipynb).

As for the MIMIC3 dataset, you must submit the application for data access from [https://mimic.physionet.org/](https://mimic.physionet.org/).
After downloading the CSVs, you first need to build the benchmark dataset according to the [https://github.com/YerevaNN/mimic3-benchmarks/](https://github.com/YerevaNN/mimic3-benchmarks/),
and then please preprocess the dataset with [preprocess_MIMIC3.ipynb](dataset/MIMIC3/preprocess_MIMIC3.ipynb).


## Training ELDA

You can run the ELDA with the jupyter file [ELDA_Train.ipynb](ELDA_Train.ipynb), and we have already set up the default setting to train the ELDA on the physionet2012 dataset.

1. Please specify the parameter for the dataset setting, such as dataset, application, data filing, standardization.
2. Please specify the parameters for the model setting, such as embed_dim, hidden_dim, clip_min, clip_mac, compress_dim.
3. Please specify the parameters for the training setting, such as batch_size, learning_rate, patience.

If you want to use your own dataset, please design the Dataloader like it in physionet2012 or MIMIC3, and do the forward imputation (i.e. impute the missing data with the last observation) and standardization before training the ELDA.
