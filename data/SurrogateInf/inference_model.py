import argparse
import os
import numpy as np
import matplotlib
import pandas as pd
from mpi4py import MPI
import csv
from collections import OrderedDict

matplotlib.use("Agg")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import horovod.keras as hvd ### importing horovod to use data parallelization in another step

from ST_funcs.smiles_regress_transformer_funcs import *
from tensorflow.python.client import device_lib
import json
from ST_funcs.smiles_pair_encoders_functions import *
import time

#######HyperParamSetting#############

json_file = 'config_inference.json'
hyper_params = ParamsJson(json_file)

######## Load model #############

model = ModelArchitecture(hyper_params).call()
model.load_weights(hyper_params['model']['weights'])

##### Set up tokenizer ########
if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
    spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

####### Iterate over files ##############
BATCH = hyper_params['general']['batch_size']

data = pd.read_csv(hyper_params['inference_data']['data'])
data_smiles = data['smiles']
maxlen = hyper_params['tokenization']['maxlen']
x_inference = preprocess_smiles_pair_encoding(data_smiles,
                                            maxlen,
                                            tokenizer)

Output = model.predict(x_inference, batch_size=BATCH)
SMILES_DS = np.vstack((data_smiles, np.array(Output).flatten())).T 
SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)

filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS).values())
filename = f'output.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['smiles', 'score'])
    writer.writerows(filtered_data)


