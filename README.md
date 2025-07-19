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

