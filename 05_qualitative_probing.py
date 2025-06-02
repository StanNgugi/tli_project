# 05_qualitative_probing.py
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
import json
import random
import os
import logging

# Import helper function from the shared utilities file
from utils import load_model

# --- Configuration ---
class ProbeConfig:
    MODEL_ID = "Lugha-Llama/Lugha-Llama-8B-wura"
    ADAPTER_PATH = "./results/tli_lora_adapters/final/" # Ensure this path is correct
    WORD_PAIRS_PATH = "./data/word_pairs.json"
    
    # Probing Parameters
    NUM_PAIRS_TO_PROBE = 7 # Number of random word pairs to select for probing
    RANDOM_STATE = 42 # For reproducibility of random sample selection

    # Generation Parameters
    MAX_NEW_TOKENS = 20 # Increased slightly for more complete answers
    TEMPERATURE = 0.1 # Keep low for consistent (less creative) outputs
    TOP_P = 0.9 # Top-p sampling to add some diversity while being focused
    DO_SAMPLE = True # Enable sampling
    
    # Output
    OUTPUT_DIR = "./results/evaluation"
    PROBE_RESULTS_PATH = os.path.join(OUTPUT_DIR, "qualitative_probe_results.md")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Script ---
def generate_response(prompt, model, tokenizer, config):
    """Generates a response from a model given a prompt."""
    model.eval() # Ensure model is in evaluation mode
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id, # Essential for Llama
            attention_mask=inputs.attention_mask # Pass attention mask
        )
    
    # Decode the generated tokens, skipping the prompt tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

def main():
    config = ProbeConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    random.seed(config.RANDOM_STATE) # Set seed for random sample selection

    logging.info("--- Phase 3: Qualitative Probing ---")

    # 1. Load Models and Tokenizer
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    logging.info("Loading base and TLI models for probing...")
    base_model = load_model(config.MODEL_ID, quant_config)
    tli_model = load_model(config.MODEL_ID, quant_config, adapter_path=config.ADAPTER_PATH)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 2. Select words and define prompt templates
    try:
        with open(config.WORD_PAIRS_PATH, 'r', encoding='utf-8') as f:
            all_pairs = json.load(f)
        logging.info(f"Loaded {len(all_pairs)} word pairs from {config.WORD_PAIRS_PATH}.")
        if not all_pairs:
            logging.error("No word pairs found for qualitative probing. Exiting.")
            return
    except FileNotFoundError:
        logging.error(f"Error: Word pairs file not found at {config.WORD_PAIRS_PATH}.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {config.WORD_PAIRS_PATH}. Check file format.")
        return

    num_pairs_to_sample = min(config.NUM_PAIRS_TO_PROBE, len(all_pairs))
    probe_pairs = random.sample(all_pairs, num_pairs_to_sample)

    # Enhanced prompt templates
    prompt_templates = [
        {"name": "Direct Translation (Sw-En)", "template": "The English word for '{swahili}' is"},
        {"name": "Reverse Translation (En-Sw)", "template": "The Swahili word for '{english}' is"},
        {"name": "Code-switched Sentence (Noun)", "template": "Nilienda sokoni kununua {swahili}. Katika Kiingereza, tunaita "},
        {"name": "Code-switched Sentence (Verb/Adj)", "template": "He is my {swahili} friend. He is very "},
        {"name": "Contextual Definition (Sw)", "template": "Swahili: {swahili}\nEnglish meaning: "},
        {"name": "Sentence Completion (Sw-En)", "template": "Nilipofika nyumbani, niliona rafiki yangu. My friend, the "}
    ]

    # 3. Generate and save responses
    logging.info(f"Generating responses for {len(probe_pairs)} pairs... Saving to {config.PROBE_RESULTS_PATH}")
    with open(config.PROBE_RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write("# Qualitative Probing Results: Base Model vs. TLI-Tuned Model\n\n")
        f.write("This file compares the outputs of the original Lugha-Llama model with the model after Targeted Lexical Injection (TLI) fine-tuning. Prompts are designed to assess direct translation and contextual understanding of targeted word pairs.\n\n")

        for pair in probe_pairs:
            sw_word, en_word = pair["swahili"], pair["english"]
            f.write(f"---\n\n## Probing for: `{sw_word}` <-> `{en_word}`\n\n")

            f.write("| Prompt Template | Prompt | Base Model Output | TLI-Tuned Model Output |\n")
            f.write("|---|---|---|---|\n")

            for p_template in prompt_templates:
                prompt_text = p_template["template"].format(swahili=sw_word, english=en_word)
                
                # Generate from base model
                base_response = generate_response(prompt_text, base_model, tokenizer, config)
                
                # Generate from TLI model
                tli_response = generate_response(prompt_text, tli_model, tokenizer, config)

                # Format outputs for Markdown table
                # Replace newlines with <br> for single line in table, escape pipe characters
                formatted_base_response = base_response.replace('\n', '<br>').replace('|', '\\|')
                formatted_tli_response = tli_response.replace('\n', '<br>').replace('|', '\\|')
                formatted_prompt_text = prompt_text.replace('|', '\\|') # Escape pipes in prompt too

                f.write(f"| {p_template['name']} | `{formatted_prompt_text}` | `{formatted_base_response}` | `{formatted_tli_response}` |\n")
            
            f.write("\n")
    
    logging.info("✅ Qualitative probing complete. Review results in Markdown file.")

if __name__ == "__main__":
    main()