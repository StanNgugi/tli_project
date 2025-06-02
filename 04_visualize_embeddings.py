# 04_visualize_embeddings.py
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
import json
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import os
import logging

# Import helper functions from the shared utilities file
from utils import load_model, get_word_embedding

# --- Configuration ---
class VisConfig:
    MODEL_ID = "Lugha-Llama/Lugha-Llama-8B-wura"
    ADAPTER_PATH = "./results/tli_lora_adapters/final/" # Ensure this path is correct
    WORD_PAIRS_PATH = "./data/word_pairs.json"
    TARGET_LAYER = 2 # 🎯 Use the same layer index as in training
    
    # Visualization Parameters
    NUM_PAIRS_TO_VISUALIZE = 25 # Keep this number manageable for a clear plot
    TSNE_PERPLEXITY = 15 # Standard range 5-50. Must be < N-1 where N is number of samples.
    TSNE_N_ITER = 1000 # Number of iterations for t-SNE optimization
    RANDOM_STATE = 42 # For reproducibility of t-SNE and random sample selection

    # Output
    OUTPUT_DIR = "./results/evaluation"
    PLOT_PRE_TLI_PATH = os.path.join(OUTPUT_DIR, "tsne_embedding_space_PRE_TLI.png")
    PLOT_POST_TLI_PATH = os.path.join(OUTPUT_DIR, "tsne_embedding_space_POST_TLI.png")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---
def main():
    config = VisConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    random.seed(config.RANDOM_STATE) # Set seed for random sample selection
    np.random.seed(config.RANDOM_STATE) # Set seed for NumPy operations (t-SNE initialization)
    
    logging.info("--- Phase 3: Embedding Space Visualization ---")

    # 1. Load Models and Tokenizer
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    logging.info("Loading base and TLI models...")
    base_model = load_model(config.MODEL_ID, quant_config)
    tli_model = load_model(config.MODEL_ID, quant_config, adapter_path=config.ADAPTER_PATH)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Data for Visualization
    try:
        with open(config.WORD_PAIRS_PATH, 'r', encoding='utf-8') as f:
            all_pairs = json.load(f)
        logging.info(f"Loaded {len(all_pairs)} word pairs from {config.WORD_PAIRS_PATH}.")
        if not all_pairs:
            logging.error("No word pairs found for visualization. Exiting.")
            return
    except FileNotFoundError:
        logging.error(f"Error: Word pairs file not found at {config.WORD_PAIRS_PATH}.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {config.WORD_PAIRS_PATH}. Check file format.")
        return
    
    # Select a random subset for clarity, ensuring we don't request more pairs than available
    num_pairs_to_sample = min(config.NUM_PAIRS_TO_VISUALIZE, len(all_pairs))
    if num_pairs_to_sample < 2: # t-SNE typically needs more than 1 point.
        logging.warning(f"Too few word pairs ({num_pairs_to_sample}) for meaningful t-SNE visualization. Skipping.")
        return

    vis_pairs = random.sample(all_pairs, num_pairs_to_sample)
    
    logging.info(f"Extracting embeddings for {len(vis_pairs)} pairs...")
    embedding_data = []
    for i, pair in enumerate(tqdm(vis_pairs, desc="Extracting embeddings")):
        sw_word, en_word = pair["swahili"], pair["english"]
        
        # Pre-TLI embeddings (L2-normalized by get_word_embedding)
        sw_emb_base = get_word_embedding(sw_word, base_model, tokenizer, config.TARGET_LAYER)
        en_emb_base = get_word_embedding(en_word, base_model, tokenizer, config.TARGET_LAYER)
        
        # Post-TLI embeddings (L2-normalized by get_word_embedding)
        sw_emb_tli = get_word_embedding(sw_word, tli_model, tokenizer, config.TARGET_LAYER)
        en_emb_tli = get_word_embedding(en_word, tli_model, tokenizer, config.TARGET_LAYER)
        
        # Only add if all embeddings were successfully extracted
        if all(e is not None for e in [sw_emb_base, en_emb_base, sw_emb_tli, en_emb_tli]):
            embedding_data.append({'pair_id': i, 'word': sw_word, 'lang': 'Swahili', 'state': 'Pre-TLI', 'embedding': sw_emb_base.numpy()})
            embedding_data.append({'pair_id': i, 'word': en_word, 'lang': 'English', 'state': 'Pre-TLI', 'embedding': en_emb_base.numpy()})
            embedding_data.append({'pair_id': i, 'word': sw_word, 'lang': 'Swahili', 'state': 'Post-TLI', 'embedding': sw_emb_tli.numpy()})
            embedding_data.append({'pair_id': i, 'word': en_word, 'lang': 'English', 'state': 'Post-TLI', 'embedding': en_emb_tli.numpy()})
        else:
            logging.warning(f"Skipping visualization for pair ({sw_word}, {en_word}) due to failed embedding extraction.")

    df = pd.DataFrame(embedding_data)

    if df.empty:
        logging.error("No valid embedding data to visualize. Exiting.")
        return

    # 3. Generate and Plot t-SNE for each state (Pre and Post)
    for state in ['Pre-TLI', 'Post-TLI']:
        logging.info(f"Generating t-SNE plot for {state} state...")
        
        state_df = df[df['state'] == state].copy()
        
        # Check if enough samples for t-SNE
        if len(state_df) < max(2, config.TSNE_PERPLEXITY + 1): # t-SNE typically needs more than 2 or (perplexity + 1) samples
            logging.warning(f"Not enough data points ({len(state_df)}) for t-SNE with perplexity {config.TSNE_PERPLEXITY} for {state} state. Skipping plot.")
            continue

        embeddings_matrix = np.vstack(state_df['embedding'].values)
        
        tsne = TSNE(
            n_components=2, 
            perplexity=min(config.TSNE_PERPLEXITY, len(state_df)-1), # Perplexity must be less than number of samples - 1
            random_state=config.RANDOM_STATE,
            n_iter=config.TSNE_N_ITER,
            # Init can be 'random' or 'pca'. 'pca' is generally faster.
            init='pca' if embeddings_matrix.shape[1] > 2 else 'random' 
        )
        tsne_results = tsne.fit_transform(embeddings_matrix)
        
        state_df['tsne-1'] = tsne_results[:,0]
        state_df['tsne-2'] = tsne_results[:,1]

        plt.figure(figsize=(16, 16))
        palette = {'Swahili': 'blue', 'English': 'red'}
        
        sns.scatterplot(
            x="tsne-1", y="tsne-2",
            hue="lang",
            palette=palette,
            data=state_df,
            s=100, # Size of points
            alpha=0.7,
            legend='full' # Show full legend
        )
        
        # Connect paired words with a line
        for pair_id in state_df['pair_id'].unique():
            pair_points = state_df[state_df['pair_id'] == pair_id].sort_values(by='lang') # Sort to ensure consistent line drawing
            if len(pair_points) == 2: # Ensure both Swahili and English points exist for the pair
                plt.plot(pair_points['tsne-1'], pair_points['tsne-2'], 'k-', alpha=0.3, linewidth=1.5, zorder=0) # Lines in background
        
        # Annotate points with the words
        for i in range(state_df.shape[0]):
            plt.text(
                x=state_df['tsne-1'].iloc[i]+0.02, # Use .iloc for positional indexing
                y=state_df['tsne-2'].iloc[i]+0.02,
                s=state_df['word'].iloc[i],
                fontdict={'color': 'black', 'size': 9, 'weight': 'bold'} # Make labels more readable
            )

        plt.title(f't-SNE Visualization of Word Embeddings ({state})', fontsize=18)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside plot
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_path = config.PLOT_PRE_TLI_PATH if state == 'Pre-TLI' else config.PLOT_POST_TLI_PATH
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logging.info(f"Plot saved to {output_path}")
        plt.close() # Close plot to free memory

if __name__ == "__main__":
    main()