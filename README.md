<h1 align="center">🧬 PrionForge.ai 🧬</h1>

<p align="center">
  <img src="https://img.shields.io/github/stars/sudhishgupta/PrionForge%2Eai?style=flat-square" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/sudhishgupta/PrionForge%2Eai?style=flat-square" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/issues/sudhishgupta/PrionForge%2Eai?style=flat-square" alt="GitHub issues"/>
  <img src="https://img.shields.io/github/license/sudhishgupta/PrionForge%2Eai?style=flat-square" alt="License"/>
</p>

<p align="center">AI‑POWERED TOOL FOR GENERATING NOVEL "PRION-LIKE" PROTEINS WITH PRECISE SUBCELLULAR TARGETTING</p>


<h2 align='left'>📋 About</h2>
<p align='left'>Prions are a fascinating, yet pathogenic class of proteins which are infectious and cause several diseases in different species of living organisms including humans. Prions unlike conventional virus like infections which occur through nucleic acids (DNA/RNA), basically infect an organism by inducing a misfolding in normal proteins(similar to them), these misfolded proteins are not degraded easily and they aggregate over time, leading to several disorders. It has been shown that these prion-proteins have one thing in common, high Q/N content in their amino-acid composition, long-stretches of Glutamine and Arginine residues in repetition, leading to formation of prion-like domains, essential for the misfolding transmission.</p>

<p align='left'>PrionForge.ai is an AI-powered tool which is digging into generation of synthetic and novel prion-like sequences, with defined subcellular location, thanks to the advent of advanced "Protein-Language Models".</p>

<h2 align='left'> Under the Hood</h2>
<h3 align='left'>🤖 Agentic Framework</h3>
<p align='left'>In order to diversify and inherit a agentic-framework for this tool, each subtask of this application was given to a separate 'protein-language-model'</p>
<img width="1918" height="606" alt="image" src="https://github.com/user-attachments/assets/e8fecd78-0c63-4fe8-a9f5-812e315bf4e8" />
<ol>
  <li><h4>Fine-Tuned ProtGPT2</h4><p>Parameters ~  774M</p>
    <p>Fine Tuned using LoRA : Low Rank Adaption for LLMs</p>
    <p>LoRA Parameters :  lora_rank = 8 & learning_rate = 1e-4</p>
    <p>Fine-Tuned over ~ 664 Prion-Sequences (from S.cerevisiae) obtained from PrionScan Database.</p>
  </li>
  <li><h4>Prot-T5-XL-UniRef50</h4>
    <p>Parameters ~ 5B</p>
    <p>Generated 1024 dimensional embeddings per protein (Prion/Non-Prion)</p>
  </li>
  <li><h4>Deeploc2.1 : Fair ESM-1b</h4>
    <p>Parameters ~ 1b</p>
    <p>Using Deeploc2.1 to predict subcellular location of the generated prion-like sequences.</p>
  </li>
</ol>

<h3>⌨️ Prion-Like Sequence Generation</h3>
<p align='left'>Prion Sequence is generated using Fine Tuned Prot-T5-XL-U50 model. More information on the base model can be found <a href=https://huggingface.co/nferruz/ProtGPT2>here</a>.
The fine-tuned model has been tested to generate sequences of upto 512 amino acids. The generated sequences have been examined and found to have high QN content (~ 25-30%), with presence of single/multiple Prion Domains, which have also been verified from existing web based tools like <a href=http://plaac.wi.mit.edu/>PLAAC</a> and <a href=http://webapps.bifi.es/prionscan>PrionScan</a>.</p>
<p align='left'>To examine the structural integrity, the generated sequences were also posted on the <a href=https://alphafoldserver.com/>AlphaFold Web Server</a>, and the structures showed high presence of alpha helices and random coils, alongwith beta-sheet motifs too, aligning with the charactertics of naturally occuring prion proteins.</p>
<img width="1585" height="525" alt="image" src="https://github.com/user-attachments/assets/6381e804-d05f-4933-b5ba-0c3b58528887" />

<h3>🗳️ Prion Classification</h3>
<p align='left'>The sequences sampled and genrated by FT-ProtGPT2 model are then filtered through a embedding based prion classifier. Protein-Language Models are known to learn deep feature representation of proteins and their functions (biological/physico-chemical) in form of numerical embeddings. To investigate if these models can distinguish between prion and non prion sequences, a preliminary test was conducted, uisng the embeddings from Prot-T5-XL-U50 model. More information on the base model can be found <a href=https://huggingface.co/Rostlab/prot_t5_xl_uniref50>here</a></p>
<img width="1776" height="1006" alt="image" src="https://github.com/user-attachments/assets/198b7ea5-12d8-4bbd-8a2b-809d8f73a5b4" />
<p>Figure (a) : The embeddings were then visualised in 2-D using t-SNE with multiple perplexity values.</p>
<p>Figure (b) : Average Q and N content in prion proteins is approximately double of that present in normal non-prion proteins.</p>
<p>Figure (c) : 3 Different models designed for <b>Prion Classification</b></p>
<p>Figure (d) : Performance Comparison of models shows that Q/N Content alone is not performing well compared to embeddings alone, but the combination of both these information, makes the model better (in terms of recall and overall F1-Score)</p>

<h3>🎯 Prion Sub-Cellular Localization Annotation</h3>
<p align='left'>For generating prion-proteins with precise subcellular location, <b>DeepLoc-2.1</b> was deployed. DeepLoc-2.1 was configured to use Fair-ESM-1b Protein Langauage Model. More information on the base model can be found <a href=https://github.com/facebookresearch/esm>here</a>.</p>
<img width="1212" height="459" alt="image" src="https://github.com/user-attachments/assets/446570f9-c0c2-4d75-ab36-2197b0a7650f" />

<h2>⚙️ Installation</h2>
To setup the environment and recommended dependencies for this tool to run on your work-station, download the files, and run the following steps.

```
python setup.py
```

<h2>🖥️ Usage</h2>
To run the tool, follow these steps :

```
conda activate PrionForge.ai
```

Then run the following command to use the tool for sample-run : 

```
python master_script.py -p NQNQ -max_L 100 -sample 20 -num 5 -batch 10 -device_dl2 cpu -sl Nucleus -classifier combined -out prion_forge_nucleus_prions.fasta 
```

To get help about the tool functioning, run the following command :
```
python master_script.py -h
```

```
usage: master_script.py [-h] [-p PROMPT] [-max_L MAX_LENGTH] [-sample SAMPLE] [-num NUM_SEQ] [-batch BATCH_SIZE]
                        [-out_pgpt2 PROTGPT2_OUTPUT] [-path PATH_TO_PROTGPT2] [-out_deeploc2 DL2_OUTPUT]
                        [-device_dl2 DEVICE_DL2] [-model_dl2 MODEL_DL2] [-sl TARGET_SL] [-classifier CLASSIFIER_TYPE]
                        [-out FINAL_OUTPUT]

Generate Novel Prion Like Proteins Sequences with Precisse Subcellular Localization

options:
  -h, --help            show this help message and exit
  -p PROMPT, --prompt PROMPT
                        Prompt for ProtGPT2 sequence generation.
  -max_L MAX_LENGTH, --max_length MAX_LENGTH
                        Maximum number of aa in generated protein sequences
  -sample SAMPLE, --sample SAMPLE
                        Number of protein sequences to be sampled at one iteration
  -num NUM_SEQ, --num_seq NUM_SEQ
                        Number of protein sequences to be generated
  -batch BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of sequences to be generated per batch of ProtGPT2 output
  -out_pgpt2 PROTGPT2_OUTPUT, --protgpt2_output PROTGPT2_OUTPUT
                        Path to write the output FASTA file.
  -path PATH_TO_PROTGPT2, --path_to_protgpt2 PATH_TO_PROTGPT2
                        Path to fine-tuned ProtGPT2 model directory.
  -out_deeploc2 DL2_OUTPUT, --dl2_output DL2_OUTPUT
                        Path to directory saving the temporary DeepLoc-2.1 Results.
  -device_dl2 DEVICE_DL2, --device_dl2 DEVICE_DL2
                        Device to be used for DeepLoc2.1
  -model_dl2 MODEL_DL2, --model_dl2 MODEL_DL2
                        Model to be used for DeepLoc2.1
  -sl TARGET_SL, --target_sl TARGET_SL
                        Target Prion Subcellular Location
  -classifier CLASSIFIER_TYPE, --classifier_type CLASSIFIER_TYPE
                        Model Type to use for Prion Classification
  -out FINAL_OUTPUT, --final_output FINAL_OUTPUT
                        Path to write the final output FASTA file.

```




