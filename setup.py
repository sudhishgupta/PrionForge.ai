#!/usr/bin/env python3
"""
setup.py

This script automates the creation of a Conda environment named 'PrionForge.ai'
with Python 3.12, installs all dependencies from requirements.txt,
and pre-downloads the Prot-T5-XL-U50 model from Hugging Face.
"""

import subprocess
import sys
import os

ENV_NAME = "PrionForge.ai"
PYTHON_VERSION = "3.12"
REQ_FILE = "requirements.txt"
HF_MODEL_ID = "Rostlab/prot_t5_xl_half_uniref50"  # Prot-T5-XL-U50

def run(cmd, **kwargs):
    """Helper to run a shell command and abort on failure."""
    print(f"> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"Error: command {' '.join(cmd)} failed with exit code {result.returncode}")

def main():
    # 1. Create the conda environment
    run([
        "conda", "create",
        "--name", ENV_NAME,
        f"python={PYTHON_VERSION}",
        "-y"
    ])

    # 2. Install dependencies via pip inside the new env
    #    Use `conda run` so we don’t need to manually activate the env in this script
    if not os.path.exists(REQ_FILE):
        sys.exit(f"Error: requirements file '{REQ_FILE}' not found in {os.getcwd()}")
    run([
        "conda", "run", "--name", ENV_NAME,
        "pip", "install", "-r", REQ_FILE
    ])

    # 3. Install Hugging Face transformers and related tooling if not already in requirements
    run([
        "conda", "run", "--name", ENV_NAME,
        "pip", "install", "transformers", "huggingface-hub"
    ])

    # 4. Pre-download the Prot-T5-XL-U50 model into the Hugging Face cache
    #    so that scripts using it won’t need to fetch it at runtime.
    python_code = (
        "from transformers import AutoTokenizer, AutoModel;\n"
        f"AutoTokenizer.from_pretrained('{HF_MODEL_ID}', cache_dir='./models/{HF_MODEL_ID}');\n"
        f"AutoModel.from_pretrained('{HF_MODEL_ID}', cache_dir='./models/{HF_MODEL_ID}');\n"
        "print('Model download complete.')"
    )
    run([
        "conda", "run", "--name", ENV_NAME,
        "python", "-c", python_code
    ])

    print(f"\n✅ Environment '{ENV_NAME}' is ready!")
    print(f"   • To activate: conda activate {ENV_NAME}")
    print("   • Models cached under ./models/")

if __name__ == "__main__":
    main()
