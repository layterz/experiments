import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from transformer_lens import HookedTransformer
from utils import *

def main(model, args):
    # Load the IMDb dataset
    imdb_dataset = load_dataset("imdb", split="train")
    # Iterate over the dataset in batches
    for i in tqdm(range(0, min(len(imdb_dataset), 99), args.batch_size)):
        batch = imdb_dataset[i:i+args.batch_size]["text"]
        batch = model.to_string(model.to_tokens(batch)[:, :args.input_size])
        
        # Call the generate function for each batch
        cache = run_prompts(model, *batch)
        batch_results = generate(cache)
        
        # Convert the batch results to a DataFrame
        batch_df = to_df(batch_results)
        # Save the results to a CSV file after each successful batch
        batch_df.to_csv(f'{args.output_file}_batch_{i}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IMDb dataset and generate results.")
    parser.add_argument("--model_name", type=str, default="gpt2-small", help="Name of the pre-trained model to use (default: 'gpt2')")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing the dataset (default: 32)")
    parser.add_argument("--input_size", type=int, default=32, help="Input size for each prompt (default: 32)")
    parser.add_argument("--output_file", type=str, default="results", help="Path to the output CSV file (default: 'results.csv')")
    args = parser.parse_args()

    # Load the pre-trained model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        args.model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

    main(model, args)