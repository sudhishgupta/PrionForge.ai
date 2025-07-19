import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def reformat_sequence(s, line_length=60):
    # Remove existing newlines
    s = s.replace('\n', '')

    # Insert a newline after every `line_length` characters
    lines = [s[i:i+line_length] for i in range(0, len(s), line_length)]
    return '\n'.join(lines)


def generate_sequences(args):
    '''
    args : argparse object : contains the required arguements
    '''

    pathToModel = fr"{args.path}"

    tokenizer = AutoTokenizer.from_pretrained(pathToModel)
    model = AutoModelForCausalLM.from_pretrained(pathToModel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    protgpt2 = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0,model_kwargs = {"max_new_tokens": int(args.max_length)})


    # Generate in batches
    num_sequences = int(args.num_seq)
    batch_size = int(args.batch_size)
    all_sequences = []

    prompt = f'<|endoftext|>M{args.prompt}'
    maxL = int(args.max_length)


    for _ in tqdm(range(num_sequences // batch_size)):
        batch = protgpt2(
            prompt,
            max_new_tokens=maxL,
            do_sample=True,
            top_k=950,
            temperature=1.2,
            repetition_penalty=1.8,
            num_return_sequences=batch_size,
            truncation=True,  # ✅ Fix tokenizer warning
            eos_token_id=0
        )
        all_sequences.extend([
            seq["generated_text"].replace("<|endoftext|>", "").strip() for seq in batch
        ])


    with open(args.output_file, 'w') as f:
        for k, seq in enumerate(all_sequences):
            f.write(f'>Sequence_{k}\n')
            f.writelines(reformat_sequence(seq))
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description="Generate protein sequences using ProtGPT2 and filter with DeepLoc2.1")

    parser.add_argument('-p','--prompt', type=str, default='NQNQ',
                        help='Prompt for ProtGPT2 sequence generation.')
    parser.add_argument('-max_L','--max_length', type=int, default=100,
                        help='Maximum number of aa in generated protein sequences')
    # parser.add_argument('--min_len', type=int, required=True,
    #                     help='Minimum allowed length for generated protein sequences.')

    parser.add_argument('-num','--num_seq',default=20,
                        help='Number of protein sequences to be generated')

    parser.add_argument('-batch','--batch_size',default=10,
                        help='Number of sequences to be generated per batch of ProtGPT2 output')

    parser.add_argument('-out_protgpt2','--output_file', type=str, default='outputProtGPT2.fasta',
                        help='Path to write the output FASTA file.')

    parser.add_argument('--path', type=str, default=r'ft_protgpt2_lora/',
                        help='Path to fine-tuned ProtGPT2 model directory.')



    args = parser.parse_args()


    generate_sequences(args)

if __name__ == '__main__':
    main()


