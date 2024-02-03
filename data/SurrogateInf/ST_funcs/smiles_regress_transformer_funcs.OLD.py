
import sys
import argparse
import os
import numpy as np
import pandas as pd
import json
from functools import partial
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
import codecs
from SmilesPE.tokenizer import *
from ST_funcs.smiles_pair_encoders_functions import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import horovod.keras as hvd ### importing horovod to use data parallelization in another step
from ST_funcs.clr_callback import *
from tensorflow.python.client import device_lib
from itertools import chain, repeat, islice
from mpi4py import MPI

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def ParamsJson(json_file):
    with open(json_file) as f:
       params = json.load(f)
    return params


def initialize_hvd():
    hvd.init() 
    print("I am rank %d of %d" %(hvd.rank(), hvd.size()))
    
    #HVD-2: GPU pinning
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    # Ping GPU to each9 rank
    for gpu in gpus:
    	tf.config.experimental.set_memory_growth(gpu,True)
    if gpus:
    	tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    return 

def initialize_mpi():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    return comm, size, rank


def split_data(data_x, data_y):
    data_x = np.array_split(data_x, hvd.size())[hvd.rank()]
    data_y = np.array_split(data_y, hvd.size())[hvd.rank()]
    return (data_x, data_y)

def large_scale_split(hyper_params, size, rank):
    DATA_FILE_PATH = hyper_params['inference_data']['data_dir']
    databases = hyper_params['inference_data']['databases']
    All_Files = np.array([])
    All_Dirs = np.array([])
    for dirs in databases:
        list_dir_files = np.array(sorted(os.listdir(f'{DATA_FILE_PATH}/{dirs}')))
        All_Files = np.concatenate((All_Files, list_dir_files))
        dir_enumerate = np.array([dirs for i in range(len(list_dir_files))]) 
        All_Dirs = np.concatenate((All_Dirs, dir_enumerate))
    
    split_files = np.array_split(All_Files, int(size/4))[int(rank/4)]
    split_dirs = np.array_split(All_Dirs, int(size/4))[int(rank/4)]

    return split_files, split_dirs

def large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank):
    DATA_FILE_PATH = hyper_params['inference_data']['data_dir']
    data_path_inference = f'{DATA_FILE_PATH}/{dirs}/{fil}'
    maxlen = hyper_params['tokenization']['maxlen']

    Data_smiles_total = pd.read_feather(data_path_inference)['SMILE']
    Data_smiles_raw = np.array_split(Data_smiles_total, 4)[rank%4]
    del(Data_smiles_total)
    
    x_inference = preprocess_smiles_pair_encoding(Data_smiles_raw,
                                                    tokenizer,
                                                    maxlen
                                                    )
    
    return Data_smiles_raw, x_inference

# Implement embedding layer
# Two seperate embedding layers, one for tokens, one for token index (positions).

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts) # turns text into tokens
    return sequence.pad_sequences(text_sequences, maxlen=max_sequence_length) # pad all sequences so they all have same length


#def train_val_data(data_path_train, data_path_vali, hvd_switch, vocab_size, maxlen):

def preprocess_smiles_pair_encoding(data, maxlen, vocab_file, spe_file):
    # some default tokens from huggingface
    
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
    #tokenized_data = np.array([list(pad(tokenizer(smi)['input_ids'], maxlen, 0)) for smi in data])
    tokenized_data = [list(pad(tokenizer(smi)['input_ids'], maxlen, 0)) for smi in data]
    return tokenized_data

def preprocess_spe_one_at_time(smiles, tokenizer, maxlen):#maxlen, vocab_file, spe_file):
    # some default tokens from huggingface
    
    #tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
    #tokenized_data = np.array([list(pad(tokenizer(smi)['input_ids'], maxlen, 0)) for smi in data])
    tokenized_data = pad(tokenizer(smiles)['input_ids'], maxlen, 0)
    #return np.asarray(list(tokenized_data))
    return list(tokenized_data)
    #return np.asarray(list(tokenized_data)).astype(np.int)


def stratified_sample(data, y, bin_left, bin_right, numbins):
    from sklearn.model_selection import train_test_split
    bins = np.linspace(bin_left, bin_right, numbins)
    y_binned = np.digitize(y, bins)

    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, stratify=y_binned)

    return x_train, x_test

if False:
    from rdkit import Chem
    from rdkit.Chem import MurckoScaffold
    
    def generate_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold
    
    def scaffold_sample():
        from collections import defaultdict
    
        scaffold_to_molecules = defaultdict(list)
    
        # Assuming you have a list of molecules with associated SMILES strings
        for molecule in molecules:
            scaffold = generate_scaffold(molecule.smiles)
            scaffold_to_molecules[scaffold].append(molecule)
        from sklearn.model_selection import train_test_split
    
        scaffolds = list(scaffold_to_molecules.keys())
        train_scaffolds, val_scaffolds = train_test_split(scaffolds, test_size=0.2)
        
        # Now, you can extract the molecules corresponding to these scaffolds
        train_molecules = [molecule for scaffold in train_scaffolds for molecule in scaffold_to_molecules[scaffold]]
        val_molecules = [molecule for scaffold in val_scaffolds for molecule in scaffold_to_molecules[scaffold]]
        return train_molecules, val_molecules



# Now scaffold_to_molecules is a dictionary where keys are scaffold SMILES
# and values are lists of molecules belonging to each scaffold.
    #import deepchem as dc
    ## creation of demo data set with some smiles strings
    #data_test= ["CC(C)Cl" , "CCC(C)CO" ,  "CCCCCCCO" , "CCCCCCCC(=O)OC" , "c3ccc2nc1ccccc1cc2c3" , "Nc2cccc3nc1ccccc1cc23" , "C1CCCCCC1" ]
    #Xs = np.zeros(len(data_test))
    #Ys = np.ones(len(data_test))
    ## creation of a deepchem dataset with the smile codes in the ids field
    #dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(data_test)),ids=data_test)
    #scaffoldsplitter = dc.splits.ScaffoldSplitter()
    #train,test = scaffoldsplitter.train_test_split(dataset)
    #return train, test

def normalize_data_0_1(data):
    min_data = min(data)
    max_data = max(data)
    
    norm_data = (data-min_data)/(max_data - min_data)#[(x - min_data)/(max_data-min_data) for x in data]
    return norm_data

def train_val_data(hyper_params):

    data_path = hyper_params['data_loading']['data_path']

    tokenizer_params = hyper_params['tokenization']['tokenizer']
    vocab_size = hyper_params['tokenization']['vocab_size']
    maxlen = hyper_params['tokenization']['maxlen']
    hvd_switch = hyper_params['general']['use_hvd']

    if hyper_params['data_loading']['stratified']:
        data = pd.read_csv(f'{data_path}')[:]
        data = data[data['DockingScore']<5]
        data = data[data['DockingScore']!=0]
        data['DockingScore'] = [-1*d for d in data['DockingScore']]#data['DockingScore'].apply(lambda x: -1*x+ 5)#np.array(-1* data['DockingScore'])+5
        data.loc[data['DockingScore'] < 0, 'DockingScore'] = 0
        #num = data._get_numeric_data()
        #num[num<0]=0

        #data['DockingScore'] = normalize_data_0_1(data['DockingScore'])
        #print(data.describe())
        y = data['DockingScore']
        bin_left = int(min(y))+1
        bin_right = int(max(y))-1
        #print(f"{bin_left}:{bin_right}")
        data_train, data_vali = stratified_sample(data, y, bin_left, bin_right, int((bin_right - bin_left)/1))
        print(data_train)
        print(data_vali)

    elif hyper_params['data_loading']['presplit']:
        data_train = pd.read_csv(f'{data_path}.train')
        data_vali = pd.read_csv(f'{data_path}.val')
     
    data_train.head()
    # Dataset has type and smiles as the two fields
    # reshaping: y formatted as [[y_1],[y_2],...] with floats
    x_smiles_train = data_train["SMILES"][1:]
    x_smiles_val = data_vali["SMILES"][1:]
    y_train = data_train["DockingScore"][1:].values.reshape(-1, 1) * 1.0 
    y_val = data_vali["DockingScore"][1:].values.reshape(-1, 1) * 1.0

    if hvd_switch:
        x_smiles_train, y_train = split_data(x_smiles_train, y_train)
    
    if tokenizer_params['category'] == 'smilespair':
        spe_file = tokenizer_params['spe_file']
        vocab_file = tokenizer_params['vocab_file']
        tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

        try:
            x_train = np.array([preprocess_spe_one_at_time(str(smi), tokenizer, maxlen) for smi in x_smiles_train.values])
            #x_smiles_train.values.apply(preprocess_spe_one_at_time, args = (tokenizer, maxlen))
            x_val = np.array([preprocess_spe_one_at_time(str(smi), tokenizer, maxlen) for smi in x_smiles_val.values])
            #x_val = x_smiles_val.values.apply(preprocess_spe_one_at_time, args = (tokenizer, maxlen))
        except:
            print(x_smiles_train)

            sys.exit()
            
        #x_val = preprocess_smiles_pair_encoding(x_smiles_val,
        #                                            maxlen,
        #                                            vocab_file,
        #                                            spe_file)
        
    else:
        tokenizer = text.Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(data_train["SMILES"])

        x_train = prep_text(data_train["SMILES"], tokenizer, maxlen)
        x_val = prep_text(data_vali["SMILES"], tokenizer, maxlen)

    return x_train, y_train, x_val, y_val


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# Implement a Transformer block as a layer
# embed_dim: number of tokens. This is used for the key_dim for the multi-head attention calculation
# ff_dim: number of nodes in Dense layer
# epsilon: needed for numerical stability... not sure what this means to be honest

class TransformerBlock(layers.Layer):
    # __init__: defining all class variables
    def __init__(self, embed_dim, num_heads, ff_dim, rate, activation, dropout1):
        super(TransformerBlock, self).__init__()
        self.drop_chck = dropout1
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)#, activation=activation)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation=activation),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    # call: building simple transformer architecture
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        if self.drop_chck:
            attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

class ModelArchitecture(layers.Layer):
    def __init__(self, hyper_params):
                
        #lr = hyper_params['general']['lr']
        vocab_size = hyper_params['tokenization']['vocab_size']
        maxlen = hyper_params['tokenization']['maxlen']
        #hvd_switch = hyper_params['general']['use_hvd']

        arch_params = hyper_params['architecture']
        embed_dim = arch_params['embedding']['embed_dim']
        num_heads = arch_params['transformer_block']['num_heads']
        ff_dim = arch_params['transformer_block']['ff_dim']
        DR_TB_1 = arch_params['transformer_block']['dr1']
        DR_TB_2 = arch_params['transformer_block']['dr2']
        DR_ff = arch_params['regressor_head']['dr']
        activation_transformer = arch_params['transformer_block']['activation']
        activation_regressor = arch_params['regressor_head']['activation']
        dropout1 = arch_params['transformer_block']['drop_mha']

        self.num_tb = arch_params['transformer_block']['num_blocks']

        self.inputs = layers.Input(shape=(maxlen,))
        self.embedding_layer = TokenAndPositionEmbedding(maxlen,
                                                        vocab_size,
                                                        embed_dim)

        self.transformer_block = TransformerBlock(embed_dim,
                                                    num_heads,
                                                    ff_dim,
                                                    DR_TB_1,
                                                    activation_transformer,
                                                    dropout1)

        self.reshape = layers.Reshape((1, maxlen * embed_dim),
                                        input_shape=(maxlen, embed_dim,))         

        self.dropout1 = layers.Dropout(DR_ff)
        self.dropout2 = layers.Dropout(DR_ff)
        self.dropout3 = layers.Dropout(DR_ff)
        self.dropout4 = layers.Dropout(DR_ff)
        self.dropout5 = layers.Dropout(DR_ff)

        self.dense1 = layers.Dense(1024, activation=activation_regressor)
        self.dense2 = layers.Dense(256, activation=activation_regressor)
        self.dense3 = layers.Dense(64, activation=activation_regressor)
        self.dense4 = layers.Dense(16, activation=activation_regressor)
        self.dense5 = layers.Dense(1, activation=activation_regressor)

        if False:
            if hvd_switch:
                lr = lr * hvd.size()
                self.opt = Adam(learning_rate=lr) 
                self.opt = hvd.DistributedOptimizer(self.opt)
            else:
                self.opt = Adam(learning_rate=lr)
    
    def call(self):
        x = self.embedding_layer(self.inputs)
        for tb in range(self.num_tb):
            x = self.transformer_block(x)

        x = self.reshape(x)

        x = self.dropout1(x)
        x = self.dense1(x)

        x = self.dropout2(x)
        x = self.dense2(x)

        x = self.dropout3(x)
        x = self.dense3(x)
        
        x = self.dropout4(x)
        x = self.dense4(x)
        
        x = self.dropout5(x)
        outputs = self.dense5(x)
        
        model = keras.Model(inputs=self.inputs, outputs=outputs)

        #model.compile(
        #    loss=self.loss_fn, optimizer=self.opt, metrics=["mae", r2]#, steps_per_execution=1000
        #)
        
        return model

class TrainingAndCallbacks:
    def __init__(self, hyper_params):
        self.hvd_switch = hyper_params['general']['use_hvd']
        checkpt_file = hyper_params['callbacks']['checkpt_file']
        csv_file = hyper_params['callbacks']['log_csv']
        patience_red_lr = hyper_params['callbacks']['patience_red_lr']
        patience_early_stop = hyper_params['callbacks']['patience_early_stop']
        lr = hyper_params['general']['lr']
        if self.hvd_switch:
            lr = lr * hvd.size()

        self.checkpointer = ModelCheckpoint(
            filepath=checkpt_file,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            )

        self.clr = CyclicLR(base_lr = 1*lr, max_lr = 5*lr, step_size=2000.)
        self.csv_logger = CSVLogger(csv_file)

        self.reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.75,
            patience=patience_red_lr,
            verbose=1,
            mode="auto",
            epsilon=0.0001,
            cooldown=3,
            min_lr=0.000000001,
            )

        self.early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            verbose=1,
            mode="auto",
            )

        if self.hvd_switch:
            #HVD broadcast initial variables from rank0 to all other processes 
            self.hvd_broadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

    def callback_defining(self):

        if self.hvd_switch:
            callbacks = [self.hvd_broadcast, self.reduce_lr, self.clr]
            if hvd.rank() == 0:
                callbacks.append(self.csv_logger)
                callbacks.append(self.early_stop)
                callbacks.append(self.checkpointer)
            return callbacks
        else:
            return [self.reduce_lr, self.clr, self.csv_logger, self.early_stop, self.checkpointer]

    def training(self, model, x_train, y_train, validation_data, hyper_params):
        BATCH = hyper_params['general']['batch_size']
        EPOCH = hyper_params['general']['epochs']

        callbacks = self.callback_defining()
        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH,
            epochs=EPOCH,
            verbose=1,
            validation_data=validation_data,
            callbacks=callbacks,
        )

        return history

def save_model(model, weights_path, model_out_path):
    model.load_weights(weights_path)
    model.save(model_out_path)
