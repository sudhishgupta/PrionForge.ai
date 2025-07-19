import warnings
warnings.filterwarnings('ignore')
import subprocess
import gc
import argparse
import sys
import os
import csv
import time
from Bio import SeqIO
import pandas as pd

def reformat_sequence(s, line_length=60):
    # Remove existing newlines
    s = s.replace('\n', '')

    # Insert a newline after every `line_length` characters
    lines = [s[i:i+line_length] for i in range(0, len(s), line_length)]
    return '\n'.join(lines)



def run_script(script,args,script_type):
    '''
    script : str : name of the script
    args : argparse object : containing neccessary args
    script_type : str : tells if its protgpt2 running or deeploc2 running

    return : outputs of the script

    '''

    if script_type == 'ProtGPT2':

        subprocess.run([
            'python', script,
            '-p', args.prompt,
            '-max_L', str(args.max_length),
            '-num', str(args.sample),
            '-batch', str(args.batch_size),
            '-o', args.protgpt2_output,
            '--path',args.path_to_protgpt2
        ],
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        
    elif script_type == "DeepLoc2":
        
        subprocess.run([
            'python',script,
            '-f', args.protgpt2_output,
            '-o', args.dl2_output,
            '-m', args.model_dl2,
            '-d', args.device_dl2
        ])

        
        
    elif script_type == "Prot_T5_Prion_Classification":
        
        result = subprocess.run([
            'python', script,
            '-i', args.protgpt2_output,
            '--classifier', args.classifier_type,
            '--per_protein', str(1)
        ],
            capture_output=True,
            text=True
        )

        output = result.stdout

        return output


    # Optional: clean RAM and GPU memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return None



def evaluate_results(args, current):
    '''
    args : argparse object : all the neccessary arguements for the program

    returns final_sequences
    '''

    target_label = args.target_sl

    dl2_output_file = f'{args.dl2_output}/results_dl2.csv'

    df = pd.read_csv(dl2_output_file)

    print(list(df.iloc[:,1]))

    final_seqs = []
    final_scores = []

    for i,pred_labels in enumerate(df.iloc[:,1]):
        if target_label.lower() in pred_labels.lower():
            final_seqs.append(df.iloc[i,0])
            final_scores.append(df.loc[i,target_label])

    sorted_pairs = dict(sorted(zip(final_seqs, final_scores), key=lambda x: x[1], reverse=True))

    final_seqs = list(sorted_pairs.keys())[:int(args.num_seq)]

    return final_seqs[:int(args.num_seq)-F]


def filter_genSeqs(selected_seq_id, fasta_file):
    '''
    selected_seq_id : list/array : containing selected sequence ids
    fasta_file : path : path to the original unfiltered protein seq file

    '''

    seqs = dict()

    for rec in SeqIO.parse(fasta_file,'fasta'):
        if str(rec.id) in selected_seq_id:
            seqs[str(rec.id)] = reformat_sequence(str(rec.seq))

    with open(fasta_file, 'w') as f:
        for k, seq in seqs.items():
            f.write(f'>{k}\n')
            f.writelines(seq)
            f.write('\n')


    return len(list(seqs.keys()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Novel Prion Like Proteins Sequences with Precisse Subcellular Localization")

    parser.add_argument('-p', '--prompt', type=str, default='NQNQ',
                        help='Prompt for ProtGPT2 sequence generation.')
    parser.add_argument('-max_L', '--max_length', type=int, default=100,
                        help='Maximum number of aa in generated protein sequences')


    parser.add_argument('-sample', '--sample', default=20,
                        help='Number of protein sequences to be sampled at one iteration')

    parser.add_argument('-num', '--num_seq', default=20,
                        help='Number of protein sequences to be generated')

    parser.add_argument('-batch', '--batch_size', default=10,
                        help='Number of sequences to be generated per batch of ProtGPT2 output')

    parser.add_argument('-out_pgpt2', '--protgpt2_output', type=str, default='outputProtGPT2.fasta',
                        help='Path to write the output FASTA file.')

    parser.add_argument('-path','--path_to_protgpt2', type=str, default=r'ProtGPT2_utils/ft_protgpt2_lora',
                        help='Path to fine-tuned ProtGPT2 model directory.')

    parser.add_argument('-out_deeploc2', '--dl2_output', type=str, default='output_dl2',
                        help='Path to directory saving the temporary DeepLoc-2.1 Results.')

    parser.add_argument('-device_dl2', '--device_dl2', type=str, default='cpu',
                        help='Device to be used for DeepLoc2.1')

    parser.add_argument('-model_dl2', '--model_dl2', type=str, default='Fast',
                        help='Model to be used for DeepLoc2.1')

    parser.add_argument('-sl', '--target_sl', type=str, default='Nucleus',
                        help='Target Prion Subcellular Location')

    parser.add_argument('-classifier', '--classifier_type', type=str, default='combined',
                        help='Model Type to use for Prion Classification')

    parser.add_argument('-out', '--final_output', type=str, default='outputPrionGPT.fasta',
                        help='Path to write the final output FASTA file.')

    args = parser.parse_args()








    F = 0
    iter = 0

    S = time.time()

    while F < int(args.num_seq):
        s_gen = time.time()
        print(f'SAMPLING ITERATION {iter+1}\nSTATS :: num_samples : {args.sample}      batch_size : {args.batch_size}\n')
        print('Now Running Fine-Tuned ProtGPT2')
        run_script('ProtGPT2_utils/ft_protgpt2.py',args,'ProtGPT2')
        e_gen = time.time()

        s_clsify = time.time()
        print('Now Running Prot-T5 Based Prion Classifier')
        op = run_script('Prion_Classifier_utils/prionClassifier.py',args,'Prot_T5_Prion_Classification')
        e_clsify = time.time()

        ## now we will edit the fasta file and only keep those sequences which are classified to be PRIONS
        f1 = filter_genSeqs(op,args.protgpt2_output)
        print(op,f1)

        s_loc = time.time()
        print('Now Running DeepLoc2 Based Subcellular Localization Assitance')
        run_script(r'\DeepLoc2_utils\deeploc2.py',args,'DeepLoc2')
        e_loc = time.time()

        #evaluate the results from DeepLoc2 and do second layer filtering
        op_dl2 = evaluate_results(args, current = F)

        f2 = filter_genSeqs(op_dl2,args.protgpt2_output)

        #print(op,op_dl2)

        del op

        #print(f'Selected Sequences {f1} ----> {f2}')

        s_app = time.time()
        ## final data apprehension
        if iter == 0:

            with open(args.final_output,'w') as file_out:
                seqs = dict()
                for rec in SeqIO.parse(args.protgpt2_output, 'fasta'):
                    seqs[str(rec.id)] = reformat_sequence(str(rec.seq))

                for k, seq in seqs.items():
                    if k in op_dl2:
                        file_out.write(f'>{k}_{iter}\n')
                        file_out.writelines(seq)
                        file_out.write('\n')

                del seqs
        else:

            with open(args.final_output, 'a') as file_out:
                seqs = dict()
                for rec in SeqIO.parse(args.protgpt2_output, 'fasta'):
                    seqs[str(rec.id)] = reformat_sequence(str(rec.seq))

                for k, seq in seqs.items():
                    if k in op_dl2:
                        file_out.write(f'>{k}_{iter}\n')
                        file_out.writelines(seq)
                        file_out.write('\n')

                del seqs



        dl2_output_file = f'{args.dl2_output}/results_dl2.csv'

        df1 = pd.read_csv(dl2_output_file)

        L = []
        for i in range(len(df1)):
            if df1.iloc[i, 0] in op_dl2 :
                L.append(list(df1.iloc[i, :]))

        del df1

        if  iter == 0:
            with open(f'{args.final_output[:-6]}_deeploc2.csv','w',newline="") as file_dl2:
                writer = csv.writer(file_dl2)
                writer.writerow(['Sequence_ID','Localizations','Signal_Peptide','Mem_Type',
                                 "Cytoplasm","Nucleus","Extracellular","Cell_membrane","Mitochondrion","Plastid",
                                 "Endoplasmic_reticulum","Lysosome/Vacuole","Golgi_apparatus","Peroxisome",
                                 "Peripheral", "Transmembrane", "Lipid anchor", "Soluble"])
                
                for j in L:
                    writer.writerow([f'{j[0]}_{iter}'] + j[1:])

        else:
            with open(f'{args.final_output[:-6]}_deeploc2.csv', 'a', newline="") as file_dl2:
                writer = csv.writer(file_dl2)
                for j in L:
                    writer.writerow([f'{j[0]}_{iter}'] + j[1:])


        e_app = time.time()

        F += f2
        iter += 1

        print(f'Iteration {iter} Finished')
        print(f'Time Taken ---->  Sequence_Generation : {int(e_gen-s_gen)}s ||||| Prion_Classification : {int(e_clsify-s_clsify)}s |||| Subcellular_Localization : {int(e_loc-s_loc)}s |||| Storing Results : {int(e_app-s_app)}s')
        print(f'{F} / {int(args.num_seq)} {args.target_sl} Prion Sequence Generated\n')


    E = time.time()

    print(f'Prion Sequence Generation Completed in {int(E-S)}s')

