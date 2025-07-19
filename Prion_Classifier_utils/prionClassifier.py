#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import argparse
import time
from pathlib import Path
import gc

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from prion_classifier_models import *

import pickle

import numpy as np

import h5py
from transformers import T5EncoderModel, T5Tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("Using device: {}".format(device))

def get_T5_model(model_dir, transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"):
    # print("Loading: {}".format(transformer_link))
    if model_dir is not None:
        pass
        # print("##########################")
        # print("Loading cached model from: {}".format(model_dir))
        # print("##########################")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
    # only cast to full-precision if no GPU is available
    if device==torch.device("cpu"):
        # print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )
    return model, vocab


def read_fasta( fasta_path ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case
                
    return sequences


def calcQN_content(seq):
    nQ = seq.count('Q')
    nN = seq.count('N')
    L = len(seq)

    return [nQ / L, nN / L]


def prepare_data(seq_dict, embed_dict):
    '''
    seq_dict : dict parsed using read_fasta function
    embed_dict : embedding_dictionary created using get_embeddings function
    '''
    QN_content = []
    embeds = []
    ID = []

    for id_,seq in seq_dict.items():
        ID.append(id_)
        QN_content.append(calcQN_content(seq))

    keys = list(embed_dict.keys())
    #print(f'Embed Keys\n{keys}')
    embeds = []
    for key in keys:
        e = embed_dict[key][:]
        embeds.append(e)

    return np.array(embeds), np.array(QN_content), ID





################################################# EMBEDDING GENERATION #################################################


def get_embeddings( seq_path,  
                   model_dir, 
                   per_protein, # whether to derive per-protein (mean-pooled) embeddings
                   max_residues=4000, # number of cumulative residues per batch
                   max_seq_len=1000, # max length after which we switch to single-sequence processing to avoid OOM
                   max_batch=100 # max number of sequences per single batch
                   ):
    
    seq_dict = dict()
    emb_dict = dict()

    # Read in fasta
    seq_dict_raw = read_fasta( seq_path )
    model, vocab = get_T5_model(model_dir)

    avg_length = sum([ len(seq) for _, seq in seq_dict_raw.items()]) / len(seq_dict_raw)
    n_long     = sum([ 1 for _, seq in seq_dict_raw.items() if len(seq)>max_seq_len])
    seq_dict   = sorted( seq_dict_raw.items(), key=lambda kv: len( seq_dict_raw[kv[0]] ), reverse=True )
    seq_dict = list(seq_dict_raw.items())

    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={}). Try lowering batch size. ".format(pdb_id, seq_len) +
                      "If single sequence processing does not work, you need more vRAM to process your protein.")
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # slice-off padded/special tokens
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                
                if per_protein:
                    emb = emb.mean(dim=0)
            
                if len(emb_dict) == 0:
                    pass
                    # print("Embedded protein {} with length {} to emb. of shape: {}".format(
                    #     identifier, s_len, emb.shape))

                emb_dict[ identifier ] = emb.detach().cpu().numpy().squeeze()

    end = time.time()
    
    # with h5py.File(str(emb_path), "w") as hf:
    #     for sequence_id, embedding in emb_dict.items():
    #         # noinspection PyUnboundLocalVariable
    #         hf.create_dataset(sequence_id, data=embedding)

    return emb_dict,seq_dict_raw

########################################################################################################################
########################################################################################################################



############################################### PRION CLASSIFICATION ###################################################

def classification(PrionModel,data,model_type,ID):
    '''
    PrionModel : pytorch model : the model built on pytorch framework
    data : torch loader object
    model_type : str : model_type  { only_embed, only_qn, combined(best) }
    ID : list : sequence IDs
    '''

    binary_predictions = []

    if model_type == 'combined':
        with torch.no_grad():
            for (input_1024, input_2) in data:  # Labels not needed for prediction
                input_1024 = input_1024.to(device)
                input_2 = input_2.to(device)

                logits = PrionModel(input_1024, input_2)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.995).float()  # Predict 0 or 1
                binary_predictions.extend(preds.cpu().numpy().astype(int).flatten().tolist())
                #print(binary_predictions)

    elif model_type == 'only_embed':
        with torch.no_grad():
            for (input_1024) in data:  # Labels not needed for prediction
                input_1024 = input_1024[0].to(device)

                logits = PrionModel(input_1024)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.998).float()  # Predict 0 or 1
                binary_predictions.extend(preds.cpu().numpy().astype(int).flatten().tolist())
                #print(binary_predictions)

    elif model_type == 'only_qn':
        with torch.no_grad():
            for (input_2) in data:  # Labels not needed for prediction
                input_2 = input_2[0].to(device)

                logits = PrionModel(input_2)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.75).float()  # Predict 0 or 1
                binary_predictions.extend(preds.cpu().numpy().astype(int).flatten().tolist())

    else:

        print('ERROR : Invalid Model Type Chose for Prion Classification !! Valid Models : {only_embed, only_qn, combined}')


    chosen_seqs = np.array(ID)[np.array(binary_predictions).astype(bool)].tolist()

    #print(binary_predictions)

    return chosen_seqs


########################################################################################################################


def preprocess_and_classify(SeqDict,EmbedDict,classifier_genre):

    E, QN, IDs = prepare_data(SeqDict,EmbedDict)

    if classifier_genre == 'combined':
        #print('In Combined')
        with open('scaler_comb_pytorch_1024.pkl', 'rb') as f1:
            scaler_1024 = pickle.load(f1)
        with open('scaler_comb_pytorch_2.pkl', 'rb') as f2:
            scaler_2 = pickle.load(f2)

        X_1024 = scaler_1024.transform(E)
        X_2 = scaler_2.transform(QN)


        model_comb = CombinedModel()
        model_comb.load_state_dict(torch.load('model_comb_weights.pth', map_location=device))
        model_comb.to(device)
        model_comb.eval()

        X_1024 = torch.from_numpy(X_1024).float()
        X_2 = torch.from_numpy(X_2).float()

        test_ds = TensorDataset(X_1024, X_2)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        qualifiedPrions = classification(PrionModel=model_comb,data=test_loader,model_type=classifier_genre,ID=IDs)

    elif classifier_genre == 'only_embed':
        #print('In Only Embed')
        with open('scaler_embed_pytorch.pkl', 'rb') as f1:
            scaler_e = pickle.load(f1)

        X_e = scaler_e.transform(E)

        model_embed = SimpleBinaryModel_onlyEmbeds()
        model_embed.load_state_dict(torch.load('model_embed_weights.pth', map_location=device))
        model_embed.to(device)
        model_embed.eval()

        X_e = torch.from_numpy(X_e).float()

        test_ds = TensorDataset(X_e)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


        qualifiedPrions = classification(PrionModel=model_embed,data=test_loader,model_type=classifier_genre,ID=IDs)


    elif classifier_genre == 'only_qn':
        print('In Only QN')
        with open('scaler_qn_based_pytorch.pkl', 'rb') as f1:
            scaler_qn = pickle.load(f1)

        X_qn = scaler_qn.transform(QN)

        model_qn = SimpleBinaryModel_onlyQN()
        model_qn.load_state_dict(torch.load('model_qn_weights.pth', map_location=device))
        model_qn.to(device)
        model_qn.eval()

        X_qn = torch.from_numpy(X_qn).float()

        test_ds = TensorDataset(X_qn)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        qualifiedPrions = classification(PrionModel=model_qn,data=test_loader,model_type=classifier_genre,ID=IDs)


    else:

        print('ERROR : Invalid Model Type Chose for Prion Classification !! Valid Models : {only_embed, only_qn, combined}')





    return qualifiedPrions


########################################################################################################################
########################################################################################################################




def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            't5_embedder.py creates T5 embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.') )
    
    # Required positional argument
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    # parser.add_argument( '-o', '--output', required=True, type=str,
    #                 help='A path for saving the created embeddings as NumPy npz file.')

    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                    default=None,
                    help='A path to a directory holding the checkpoint for a pre-trained model' )

    parser.add_argument('--classifier', required=False, type=str,
                    default='combined',
                    help='Type of Prion Classifier to be Used. { only_embed, only_qn, combined(best) }' )

    # Optional argument
    parser.add_argument('--per_protein', type=int, 
                    default=0,
                    help="Whether to return per-residue embeddings (0: default) or the mean-pooled per-protein representation (1).")
    return parser

def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()
    
    seq_path   = Path( args.input )

    classifier = args.classifier
    model_dir  = Path( args.model ) if args.model is not None else None

    per_protein = False if int(args.per_protein)==0 else True

    #***********************************************************************************************#
    EMBEDDING_dict,SEQUENCE_dict = get_embeddings( seq_path, model_dir, per_protein=per_protein )
    # ***********************************************************************************************#

    #print(list(SEQUENCE_dict.keys()))

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    #print('Generated Embeddings')

    #************************************************************************************************************************#
    qualified_prions = preprocess_and_classify(SeqDict=SEQUENCE_dict, EmbedDict=EMBEDDING_dict, classifier_genre=classifier)
    # ************************************************************************************************************************#

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return qualified_prions


if __name__ == '__main__':
    start = time.time()
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    qp = main()
    end = time.time()
    print(qp)
    #print(end-start)
