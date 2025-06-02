# 03_quantitative_evaluation.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
import json
import pandas as pd
from scipy.stats import ttest_rel
from tqdm import tqdm
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
class EvalConfig:
    MODEL_ID = "Lugha-Llama/Lugha-Llama-8B-wura"
    # 🎯 Path to your trained LoRA adapters
    ADAPTER_PATH = "./results/tli_lora_adapters/final/" # Ensure this path is correct
    
    # Data paths
    TRAINED_PAIRS_PATH = "./data/word_pairs.json"
    CONTROL_PAIRS_PATH = "./data/control_word_pairs.json" # Unseen pairs, ~50-100 pairs
    
    # From Phase 2 training
    TARGET_LAYER = 2 # 🎯 IMPORTANT: Use the same layer index as determined in 01_pilot_layer_selection.py
    
    # Output
    OUTPUT_DIR = "./results/evaluation"
    RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "quantitative_results.csv")

# --- Helper Functions (Copied and adapted from 02_train_tli.py for self-contained evaluation) ---

def load_model(model_id, quantization_config, adapter_path=None):
    """
    Loads the base model, and optionally merges LoRA adapters if adapter_path is provided.
    Handles rope_scaling config patching and sets pad_token_id.
    """
    logging.info(f"Loading model: {model_id} (Adapters: {adapter_path if adapter_path else 'None'})")

    # Load the configuration first
    model_config = AutoConfig.from_pretrained(model_id)

    # Apply the rope_scaling patch if necessary
    if hasattr(model_config, "rope_scaling") and isinstance(model_config.rope_scaling, dict):
        if "type" not in model_config.rope_scaling:
            logging.warning("`rope_scaling` in model config is missing 'type' key. Patching to 'linear' as a workaround.")
            model_config.rope_scaling["type"] = "linear" # Default to 'linear' if 'type' is missing
        
        # Ensure 'factor' is also present, as it's typically required with 'type'
        if "factor" not in model_config.rope_scaling:
            logging.warning("`rope_scaling` in model config is missing 'factor' key. Patching to 1.0 as a workaround.")
            model_config.rope_scaling["factor"] = 1.0 # Default factor if missing

        logging.info(f"Final `rope_scaling` config after potential patch: {model_config.rope_scaling}")
    else:
        logging.info("`rope_scaling` attribute not found or not a dictionary in model config. No patch applied.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config, # Pass the potentially modified config
        quantization_config=quantization_config,
        device_map="auto"
    )

    if adapter_path:
        logging.info(f"Loading and merging LoRA adapters from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload() # Merge LoRA weights into the base model
        logging.info("LoRA adapters merged successfully.")

    # Tokenizer must be loaded separately as it's not part of model.from_pretrained's output
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id # Ensure model config also reflects this

    model.eval() # Set model to evaluation mode
    return model, tokenizer

def get_word_embedding(word, model, tokenizer, layer_idx):
    """
    Pass a single word through the model and get its mean-pooled embedding from the target layer.
    Embeddings are L2 normalized before returning.
    """
    # Tokenize the single word, adding special tokens if needed (e.g., bos_token, eos_token)
    # The training process implicitly used these by padding=True
    # For a single word, it's often good to explicitly add special tokens if it's how they were trained.
    # However, for getting word embeddings, we usually want the representation of just the word.
    # Let's try without special tokens first, or with just `add_special_tokens=False` if we want raw word embeddings.
    # For now, stick to the training tokenization style, which included default special tokens.
    
    tokens = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings)
    
    # Ensure tokens are on the correct device
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    
    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        outputs = model(**tokens, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer_idx] # (batch_size=1, sequence_length, hidden_size)
    
    # Calculate the number of non-padding tokens. This is crucial for mean pooling.
    # num_actual_tokens_per_sequence will be (batch_size=1, 1)
    num_actual_tokens_per_sequence = tokens['attention_mask'].sum(dim=1, keepdim=True) 
    num_actual_tokens_per_sequence = torch.clamp(num_actual_tokens_per_sequence, min=1e-9) # Avoid division by zero
    
    # Expand attention mask to match hidden_states dimensions for element-wise masking
    attention_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
    masked_hidden_states = hidden_states * attention_mask_expanded
    
    # Sum along the sequence dimension (dim=1)
    sum_embeddings = torch.sum(masked_hidden_states, dim=1) # Shape: (batch_size, hidden_size)
    
    # Mean pool by dividing sum by number of tokens
    pooled_embedding = sum_embeddings / num_actual_tokens_per_sequence
    
    # L2 Normalize embeddings before returning
    normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)
    
    return normalized_embedding.squeeze(0) # Remove batch dimension for a single embedding (D,)

# --- Evaluation Logic ---

def evaluate_pairs(word_pairs, base_model, tli_model, tokenizer, config):
    """
    Calculates pre- and post-TLI similarities for a list of word pairs.
    Uses L2-normalized embeddings from get_word_embedding.
    """
    results = []
    logging.info(f"Evaluating {len(word_pairs)} pairs...")
    
    for pair in tqdm(word_pairs, desc="Evaluating pairs"):
        sw_word, en_word = pair["swahili"], pair["english"]
        
        # Get embeddings from BASE model
        # The tokenizer is passed to get_word_embedding
        sw_emb_base = get_word_embedding(sw_word, base_model, tokenizer, config.TARGET_LAYER)
        en_emb_base = get_word_embedding(en_word, base_model, tokenizer, config.TARGET_LAYER)
        
        # Get embeddings from TLI-TUNED model
        sw_emb_tli = get_word_embedding(sw_word, tli_model, tokenizer, config.TARGET_LAYER)
        en_emb_tli = get_word_embedding(en_word, tli_model, tokenizer, config.TARGET_LAYER)

        # Ensure embeddings were successfully retrieved (check if not None, although get_word_embedding should always return a tensor now)
        if sw_emb_base is None or en_emb_base is None or sw_emb_tli is None or en_emb_tli is None:
            logging.warning(f"Skipping pair ({sw_word}, {en_word}) due to failed embedding extraction.")
            continue

        # Calculate cosine similarities (embeddings are already normalized, so matmul is efficient)
        sim_pre_tli = torch.dot(sw_emb_base, en_emb_base).item()
        sim_post_tli = torch.dot(sw_emb_tli, en_emb_tli).item()
        
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
    # Use a small epsilon to avoid math errors if pre-TLI sim is exactly 0
    pct_improvement = (abs_change / (abs(mean_pre) + 1e-9)) * 100 
    
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
    
    # 1. Load Models and Tokenizer
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    # Load base model and its tokenizer
    base_model, tokenizer_base = load_model(config.MODEL_ID, quant_config)
    # Load TLI-tuned model (which merges adapters) and its tokenizer
    tli_model, tokenizer_tli = load_model(config.MODEL_ID, quant_config, adapter_path=config.ADAPTER_PATH)

    # IMPORTANT: Ensure both models use the same tokenizer instance for consistency in tokenization
    # Since load_model returns a tokenizer, we'll ensure they are identical for evaluation.
    # In this case, `load_model` always re-initializes, so `tokenizer_base` and `tokenizer_tli` will be identical.
    # We'll just use `tokenizer_tli` for all calls to avoid confusion.
    tokenizer = tokenizer_tli # Use the tokenizer loaded with the TLI model (they should be identical)

    # 2. Load Datasets
    try:
        with open(config.TRAINED_PAIRS_PATH, 'r', encoding='utf-8') as f:
            trained_pairs = json.load(f)
        logging.info(f"Loaded {len(trained_pairs)} trained word pairs from {config.TRAINED_PAIRS_PATH}.")
        if not trained_pairs:
            logging.warning(f"No trained word pairs found in {config.TRAINED_PAIRS_PATH}. Evaluation on this group will be skipped.")
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
        if not control_pairs:
            logging.warning(f"No control word pairs found in {config.CONTROL_PAIRS_PATH}. Evaluation on this group will be skipped.")
    except FileNotFoundError:
        logging.error(f"Error: Control pairs file not found at {config.CONTROL_PAIRS_PATH}. Please create 'control_word_pairs.json' with 50-100 unseen pairs.")
        # We don't return here, so evaluation can proceed on trained pairs if control is missing
        control_pairs = [] # Set to empty list to avoid further errors
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {config.CONTROL_PAIRS_PATH}. Check file format.")
        control_pairs = [] # Set to empty list to avoid further errors
        
    # 3. Evaluate and Analyze
    # Pass the shared tokenizer instance
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