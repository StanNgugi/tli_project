# 04_visualize_embeddings.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
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

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# --- Helper Functions (Copied directly for self-contained execution) ---

def load_model(model_id, quantization_config, adapter_path=None):
    """
    Loads the base model, and optionally merges LoRA adapters if adapter_path is provided.
    Handles rope_scaling config patching and sets pad_token_id.
    """
    logging.info(f"Loading model: {model_id} (Adapters: {adapter_path if adapter_path else 'None'})")

    model_config = AutoConfig.from_pretrained(model_id)

    if hasattr(model_config, "rope_scaling") and isinstance(model_config.rope_scaling, dict):
        if "type" not in model_config.rope_scaling:
            logging.warning("`rope_scaling` in model config is missing 'type' key. Patching to 'linear' as a workaround.")
            model_config.rope_scaling["type"] = "linear"
        if "factor" not in model_config.rope_scaling:
            logging.warning("`rope_scaling` in model config is missing 'factor' key. Patching to 1.0 as a workaround.")
            model_config.rope_scaling["factor"] = 1.0
        logging.info(f"Final `rope_scaling` config after potential patch: {model_config.rope_scaling}")
    else:
        logging.info("`rope_scaling` attribute not found or not a dictionary in model config. No patch applied.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        quantization_config=quantization_config,
        device_map="auto"
    )

    if adapter_path:
        logging.info(f"Loading and merging LoRA adapters from {adapter_path}...")
        if not os.path.exists(adapter_path):
            logging.error(f"Error: LoRA adapter path '{adapter_path}' does not exist.")
            logging.warning("Proceeding with base model as adapters could not be loaded/merged.")
            adapter_path = None # Mark as not loaded
        else:
            try:
                model = PeftModel.from_pretrained(model, adapter_path)
                model = model.merge_and_unload()
                logging.info("LoRA adapters merged successfully.")
            except Exception as e:
                logging.error(f"Error merging LoRA adapters from '{adapter_path}': {e}")
                logging.warning("Proceeding with base model as adapters could not be loaded/merged.")
                adapter_path = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer

def get_word_embedding(word, model, tokenizer, layer_idx):
    """
    Pass a single word through the model and get its mean-pooled embedding from the target layer.
    Excludes special tokens from pooling for a more accurate word embedding.
    Embeddings are L2 normalized before returning.
    """
    tokens = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer_idx]
    
    pooling_mask = tokens['attention_mask'].clone().float() # (1, sequence_length)
    
    for i, token_id in enumerate(tokens['input_ids'][0]):
        if token_id == tokenizer.bos_token_id or token_id == tokenizer.eos_token_id or token_id == tokenizer.pad_token_id:
            pooling_mask[0, i] = 0.0
    
    pooling_mask_expanded = pooling_mask.unsqueeze(-1).expand(hidden_states.size())
    masked_hidden_states = hidden_states * pooling_mask_expanded
    
    sum_embeddings = torch.sum(masked_hidden_states, dim=1)
    
    num_contributing_tokens = pooling_mask.sum(dim=1, keepdim=True)
    num_contributing_tokens = torch.clamp(num_contributing_tokens, min=1e-9)
    
    pooled_embedding = sum_embeddings / num_contributing_tokens
    
    # L2 Normalize embeddings before returning
    normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)
    
    return normalized_embedding.squeeze(0) # Remove batch dimension for a single embedding (D,)


# --- Main Script ---
def main():
    config = VisConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    random.seed(config.RANDOM_STATE) # Set seed for random sample selection
    np.random.seed(config.RANDOM_STATE) # Set seed for NumPy operations (t-SNE initialization)
    
    logging.info("--- Phase 4: Embedding Space Visualization ---")

    # 1. Load Models and Tokenizer
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    logging.info("Loading base model...")
    base_model, tokenizer = load_model(config.MODEL_ID, quant_config) # Use tokenizer from base load
    logging.info("Loading TLI-tuned model...")
    tli_model, _ = load_model(config.MODEL_ID, quant_config, adapter_path=config.ADAPTER_PATH) # No need for second tokenizer

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
        
        # Only add if all embeddings were successfully extracted and are not zero-norm
        if all(e is not None for e in [sw_emb_base, en_emb_base, sw_emb_tli, en_emb_tli]) and \
           torch.norm(sw_emb_base).item() > 1e-6 and torch.norm(en_emb_base).item() > 1e-6 and \
           torch.norm(sw_emb_tli).item() > 1e-6 and torch.norm(en_emb_tli).item() > 1e-6:
            
            embedding_data.append({'pair_id': i, 'word': sw_word, 'lang': 'Swahili', 'state': 'Pre-TLI', 'embedding': sw_emb_base.cpu().numpy()})
            embedding_data.append({'pair_id': i, 'word': en_word, 'lang': 'English', 'state': 'Pre-TLI', 'embedding': en_emb_base.cpu().numpy()})
            embedding_data.append({'pair_id': i, 'word': sw_word, 'lang': 'Swahili', 'state': 'Post-TLI', 'embedding': sw_emb_tli.cpu().numpy()})
            embedding_data.append({'pair_id': i, 'word': en_word, 'lang': 'English', 'state': 'Post-TLI', 'embedding': en_emb_tli.cpu().numpy()})
        else:
            logging.warning(f"Skipping visualization for pair ({sw_word}, {en_word}) due to failed or zero-norm embedding extraction.")

    df = pd.DataFrame(embedding_data)

    if df.empty:
        logging.error("No valid embedding data to visualize after filtering. Exiting.")
        return

    # 3. Generate and Plot t-SNE for each state (Pre and Post)
    for state in ['Pre-TLI', 'Post-TLI']:
        logging.info(f"Generating t-SNE plot for {state} state...")
        
        state_df = df[df['state'] == state].copy()
        
        # Check if enough samples for t-SNE, considering perplexity requirement
        # Perplexity must be less than the number of samples
        n_samples = len(state_df)
        if n_samples < 2:
            logging.warning(f"Not enough data points ({n_samples}) for t-SNE for {state} state. Skipping plot.")
            continue
        
        # Adjust perplexity if necessary
        current_perplexity = min(config.TSNE_PERPLEXITY, n_samples - 1)
        if current_perplexity < 1: # Perplexity must be at least 1 (or often > 1 for good results)
             logging.warning(f"Perplexity became too small ({current_perplexity}) for t-SNE for {state} state. Skipping plot.")
             continue


        embeddings_matrix = np.vstack(state_df['embedding'].values)
        
        tsne = TSNE(
            n_components=2, 
            perplexity=current_perplexity,
            random_state=config.RANDOM_STATE,
            n_iter=config.TSNE_N_ITER,
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
            s=100,
            alpha=0.7,
            legend='full'
        )
        
        # Connect paired words with a line
        for pair_id in state_df['pair_id'].unique():
            pair_points = state_df[state_df['pair_id'] == pair_id].sort_values(by='lang')
            if len(pair_points) == 2:
                plt.plot(pair_points['tsne-1'], pair_points['tsne-2'], 'k-', alpha=0.3, linewidth=1.5, zorder=0)
        
        # Annotate points with the words
        for i in range(state_df.shape[0]):
            plt.text(
                x=state_df['tsne-1'].iloc[i]+0.02,
                y=state_df['tsne-2'].iloc[i]+0.02,
                s=state_df['word'].iloc[i],
                fontdict={'color': 'black', 'size': 9, 'weight': 'bold'}
            )

        plt.title(f't-SNE Visualization of Word Embeddings ({state})', fontsize=18)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        output_path = config.PLOT_PRE_TLI_PATH if state == 'Pre-TLI' else config.PLOT_POST_TLI_PATH
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logging.info(f"Plot saved to {output_path}")
        plt.close()

if __name__ == "__main__":
    main()