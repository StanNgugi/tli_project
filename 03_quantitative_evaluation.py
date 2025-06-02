# 03_quantitative_evaluation.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig
import json
import pandas as pd
from scipy.stats import ttest_rel
from tqdm import tqdm
import os
import logging

# Import helper functions from the shared utilities file
from utils import load_model, get_word_embedding 

# --- Configuration ---
class EvalConfig:
    MODEL_ID = "Lugha-Llama/Lugha-Llama-8B-wura"
    # 🎯 Path to your trained LoRA adapters
    ADAPTER_PATH = "./results/tli_lora_adapters/final/" # Ensure this path is correct
    
    # Data paths
    TRAINED_PAIRS_PATH = "./data/word_pairs.json"
    CONTROL_PAIRS_PATH = "./data/control_word_pairs.json" # Unseen pairs, ~50-100 pairs
    
    # From Phase 2 training
    TARGET_LAYER = 18 # 🎯 IMPORTANT: Use the same layer index as determined in 01_pilot_layer_selection.py
    
    # Output
    OUTPUT_DIR = "./results/evaluation"
    RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "quantitative_results.csv")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (specific to this script) ---
def evaluate_pairs(word_pairs, base_model, tli_model, tokenizer, config):
    """
    Calculates pre- and post-TLI similarities for a list of word pairs.
    Uses L2-normalized embeddings from get_word_embedding.
    """
    results = []
    logging.info(f"Evaluating {len(word_pairs)} pairs...")
    
    for pair in tqdm(word_pairs, desc="Evaluating pairs"):
        sw_word, en_word = pair["swahili"], pair["english"]
        
        # Get embeddings from BASE model (already L2-normalized by get_word_embedding)
        sw_emb_base = get_word_embedding(sw_word, base_model, tokenizer, config.TARGET_LAYER)
        en_emb_base = get_word_embedding(en_word, base_model, tokenizer, config.TARGET_LAYER)
        
        # Get embeddings from TLI-TUNED model (already L2-normalized by get_word_embedding)
        sw_emb_tli = get_word_embedding(sw_word, tli_model, tokenizer, config.TARGET_LAYER)
        en_emb_tli = get_word_embedding(en_word, tli_model, tokenizer, config.TARGET_LAYER)

        # Ensure embeddings were successfully retrieved
        if sw_emb_base is None or en_emb_base is None or sw_emb_tli is None or en_emb_tli is None:
            logging.warning(f"Skipping pair ({sw_word}, {en_word}) due to failed embedding extraction.")
            continue

        # Calculate cosine similarities (embeddings are already normalized)
        sim_pre_tli = F.cosine_similarity(sw_emb_base.unsqueeze(0), en_emb_base.unsqueeze(0)).item()
        sim_post_tli = F.cosine_similarity(sw_emb_tli.unsqueeze(0), en_emb_tli.unsqueeze(0)).item()
        
        results.append({
            "swahili": sw_word,
            "english": en_word,
            "sim_pre_tli": sim_pre_tli,
            "sim_post_tli": sim_post_tli
        })
    return pd.DataFrame(results)

def analyze_and_print_results(df, group_name):
    """
    Performs analysis (mean, std, improvement, paired t-test) and prints a summary table.
    """
    if df.empty:
        logging.info(f"\nNo data to analyze for: {group_name.upper()} WORD PAIRS. Skipping analysis.")
        print("\n" + "="*50)
        print(f"ANALYSIS FOR: {group_name.upper()} WORD PAIRS (No data)")
        print("="*50 + "\n")
        return

    df['improvement'] = df['sim_post_tli'] - df['sim_pre_tli']
    
    mean_pre = df['sim_pre_tli'].mean()
    mean_post = df['sim_post_tli'].mean()
    std_pre = df['sim_pre_tli'].std()
    std_post = df['sim_post_tli'].std()
    
    abs_change = mean_post - mean_pre
    # Handle division by zero for percentage improvement if mean_pre is close to 0
    pct_improvement = (abs_change / abs(mean_pre)) * 100 if abs(mean_pre) > 1e-9 else float('inf') # Use a small epsilon
    
    # Paired t-test for statistical significance
    # ttest_rel requires at least 2 samples, handle gracefully if not enough data
    if len(df) >= 2:
        ttest_result = ttest_rel(df['sim_post_tli'], df['sim_pre_tli'])
        t_statistic = ttest_result.statistic
        p_value = ttest_result.pvalue
    else:
        t_statistic = float('nan')
        p_value = float('nan')
        logging.warning(f"Not enough data ({len(df)} samples) for paired t-test for {group_name} group.")

    print("\n" + "="*50)
    print(f"ANALYSIS FOR: {group_name.upper()} WORD PAIRS")
    print("="*50)
    print(f"Mean Similarity (Pre-TLI):  {mean_pre:.4f} (Std: {std_pre:.4f})")
    print(f"Mean Similarity (Post-TLI): {mean_post:.4f} (Std: {std_post:.4f})")
    print("-" * 50)
    print(f"Absolute Improvement:         {abs_change:+.4f}")
    print(f"Percentage Improvement:       {pct_improvement:+.2f}%")
    print("-" * 50)
    print("Paired t-test results:")
    print(f"  - T-statistic: {t_statistic:.4f}")
    print(f"  - p-value:     {p_value:.2e}")
    print("="*50 + "\n")

# --- Main Script ---
def main():
    config = EvalConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    logging.info("--- Phase 3: Quantitative Evaluation ---")
    
    # 1. Load Models
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    # Load base model
    base_model = load_model(config.MODEL_ID, quant_config)
    # Load TLI-tuned model (which merges adapters)
    tli_model = load_model(config.MODEL_ID, quant_config, adapter_path=config.ADAPTER_PATH)

    # Re-initialize tokenizer after models are loaded, to ensure pad_token_id is set
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Datasets
    try:
        with open(config.TRAINED_PAIRS_PATH, 'r', encoding='utf-8') as f:
            trained_pairs = json.load(f)
        logging.info(f"Loaded {len(trained_pairs)} trained word pairs from {config.TRAINED_PAIRS_PATH}.")
    except FileNotFoundError:
        logging.error(f"Error: Trained pairs file not found at {config.TRAINED_PAIRS_PATH}. Please check path.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {config.TRAINED_PAIRS_PATH}. Check file format.")
        return

    try:
        with open(config.CONTROL_PAIRS_PATH, 'r', encoding='utf-8') as f:
            control_pairs = json.load(f)
        logging.info(f"Loaded {len(control_pairs)} control word pairs from {config.CONTROL_PAIRS_PATH}.")
    except FileNotFoundError:
        logging.error(f"Error: Control pairs file not found at {config.CONTROL_PAIRS_PATH}. Please create 'control_word_pairs.json' with 50-100 unseen pairs.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {config.CONTROL_PAIRS_PATH}. Check file format.")
        return
        
    # 3. Evaluate and Analyze
    trained_results_df = evaluate_pairs(trained_pairs, base_model, tli_model, tokenizer, config)
    trained_results_df['group'] = 'trained'
    
    control_results_df = evaluate_pairs(control_pairs, base_model, tli_model, tokenizer, config)
    control_results_df['group'] = 'control'
    
    # 4. Print Summaries
    analyze_and_print_results(trained_results_df, "Trained")
    analyze_and_print_results(control_results_df, "Control (Unseen)")

    # 5. Save detailed results to CSV
    all_results_df = pd.concat([trained_results_df, control_results_df], ignore_index=True)
    all_results_df.to_csv(config.RESULTS_CSV_PATH, index=False)
    logging.info(f"Detailed results saved to {config.RESULTS_CSV_PATH}")

if __name__ == "__main__":
    main()