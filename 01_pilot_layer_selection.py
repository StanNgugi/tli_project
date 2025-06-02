# 01_pilot_layer_selection.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging

# --- Configuration ---
MODEL_ID = "Lugha-Llama/Lugha-Llama-8B-wura"
WORD_PAIRS_PATH = "./data/word_pairs.json"
OUTPUT_DIR = "./results/pilot_study"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---
def get_word_embedding(text, model, tokenizer, layer_idx):
    """
    Extracts the mean-pooled embedding for a word from a specific layer.
    This version correctly handles potential padding and special tokens
    by using the attention mask for mean pooling.
    """
    model.eval() # Ensure model is in evaluation mode for consistent embedding extraction
    
    # Tokenize with padding=True, truncation=True, even for single words,
    # to ensure attention_mask is generated and consistent with batching logic.
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer_idx] # Shape: (batch_size, sequence_length, hidden_size)
    attention_mask = tokens['attention_mask'].unsqueeze(-1).expand(hidden_states.size()) # Shape: (batch_size, sequence_length, hidden_size)
    
    # Mask out padding tokens (their hidden states become 0)
    masked_hidden_states = hidden_states * attention_mask
    
    # Sum embeddings of non-padding tokens
    sum_embeddings = torch.sum(masked_hidden_states, dim=1) # Shape: (batch_size, hidden_size)
    
    # Count number of non-padding tokens (excluding BOS if present, though for single words it's usually 1 + BOS)
    # The attention mask automatically handles BOS if it's considered an active token.
    num_non_padding = attention_mask.sum(dim=1) # Shape: (batch_size, 1)
    
    # Avoid division by zero for empty sequences (though unlikely with padding=True)
    num_non_padding = torch.clamp(num_non_padding, min=1e-9)
    
    # Mean pool by dividing sum by count
    embedding = sum_embeddings / num_non_padding
    
    # For a single word input, embedding will be (1, hidden_size)
    return embedding.squeeze(0) # Remove batch dimension for single word embedding

def main():
    """
    Main function to run the pilot layer selection analysis.
    """
    logging.info("--- TLI Pilot Layer Selection ---")
    logging.info(f"Using device: {DEVICE}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Model and Tokenizer (Quantized)
    logging.info(f"Loading base model: {MODEL_ID}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto" # Automatically distributes across GPUs if available
    )
    # Llama tokenizer does not have a default pad token, usually uses EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # It's good practice to set this on the model's config too, though less critical for embedding extraction
        model.config.pad_token_id = tokenizer.eos_token_id
        
    logging.info(f"Model loaded with {model.config.num_hidden_layers} layers.")

    # 2. Load Curated Word Pairs
    logging.info(f"Loading word pairs from: {WORD_PAIRS_PATH}")
    try:
        with open(WORD_PAIRS_PATH, 'r', encoding='utf-8') as f:
            word_pairs = json.load(f)
        logging.info(f"Loaded {len(word_pairs)} word pairs.")
        if not word_pairs:
            logging.error("No word pairs found. Please ensure word_pairs.json is correctly populated.")
            return
    except FileNotFoundError:
        logging.error(f"Error: word_pairs.json not found at {WORD_PAIRS_PATH}. Please create the 'data' directory and place the file there.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {WORD_PAIRS_PATH}. Check file format.")
        return

    # 3. Iterate through layers and calculate similarity
    num_layers = model.config.num_hidden_layers
    layer_similarities = []

    logging.info(f"Analyzing {num_layers} layers. This may take a while...")
    for i in tqdm(range(num_layers), desc="Processing Layers"):
        similarities_for_layer = []
        for pair in word_pairs:
            sw_word = pair["swahili"]
            en_word = pair["english"]
            
            try:
                sw_emb = get_word_embedding(sw_word, model, tokenizer, layer_idx=i)
                en_emb = get_word_embedding(en_word, model, tokenizer, layer_idx=i)
                
                # Embeddings are already normalized in get_word_embedding (implicitly by mean pooling)
                # For cosine similarity, explicit L2 normalization is standard and crucial.
                sw_emb = F.normalize(sw_emb, p=2, dim=0) # Normalize along the embedding dimension
                en_emb = F.normalize(en_emb, p=2, dim=0) # Normalize along the embedding dimension

                similarity = F.cosine_similarity(sw_emb.unsqueeze(0), en_emb.unsqueeze(0))
                similarities_for_layer.append(similarity.item())
            except Exception as e:
                logging.warning(f"Skipping pair ({sw_word}, {en_word}) at layer {i} due to error: {e}")
                continue

        avg_similarity = sum(similarities_for_layer) / len(similarities_for_layer) if similarities_for_layer else 0
        layer_similarities.append({"layer": i, "avg_cosine_similarity": avg_similarity})
        logging.info(f"Layer {i}: Avg. Similarity = {avg_similarity:.4f}")

    # 4. Analyze Results and Plot
    results_df = pd.DataFrame(layer_similarities)
    
    if results_df.empty:
        logging.error("No similarity results generated. Cannot determine best layer or plot.")
        return

    best_layer_row = results_df.loc[results_df['avg_cosine_similarity'].idxmax()]
    best_layer = int(best_layer_row['layer'])
    max_sim = best_layer_row['avg_cosine_similarity']
    
    logging.info("\n--- Analysis Complete ---")
    logging.info(f"✅ Best performing layer: {best_layer} (Avg. Cosine Similarity: {max_sim:.4f})")
    logging.info("This layer index will be used for TLI training.")

    # Plotting
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")
    plot = sns.barplot(data=results_df, x='layer', y='avg_cosine_similarity', color='skyblue')
    plot.axvline(x=best_layer, color='r', linestyle='--', label=f'Best Layer ({best_layer})')
    plt.title('Pre-existing Lexical Alignment per Layer in Lugha-Llama', fontsize=16)
    plt.xlabel('Model Layer Index', fontsize=12)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "layer_alignment_analysis.png")
    plt.savefig(plot_path)
    logging.info(f"📈 Results plot saved to: {plot_path}")

    # Optionally, save results to CSV
    results_csv_path = os.path.join(OUTPUT_DIR, "layer_alignment_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Raw results saved to: {results_csv_path}")

if __name__ == "__main__":
    main()