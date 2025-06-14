# 05_qualitative_probing.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
import json
import random
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# --- Helper Functions (Self-contained for robustness and consistency) ---

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
            adapter_path = None
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

# --- Main Script ---
def generate_response(prompt, model, tokenizer, config):
    """Generates a response from a model given a prompt."""
    model.eval() # Ensure model is in evaluation mode
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, # This correctly unpacks 'input_ids' and 'attention_mask'
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id, # Essential for Llama
            # Removed redundant 'attention_mask=inputs.attention_mask'
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

def main():
    config = ProbeConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    random.seed(config.RANDOM_STATE)

    logging.info("--- Phase 5: Qualitative Probing ---")

    # 1. Load Models and Tokenizer
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    logging.info("Loading base model for probing...")
    base_model, tokenizer = load_model(config.MODEL_ID, quant_config) 
    
    logging.info("Loading TLI-tuned model for probing...")
    tli_model, _ = load_model(config.MODEL_ID, quant_config, adapter_path=config.ADAPTER_PATH)

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
                
                base_response = generate_response(prompt_text, base_model, tokenizer, config)
                tli_response = generate_response(prompt_text, tli_model, tokenizer, config)

                formatted_base_response = base_response.replace('\n', '<br>').replace('|', '\\|')
                formatted_tli_response = tli_response.replace('\n', '<br>').replace('|', '\\|')
                formatted_prompt_text = prompt_text.replace('|', '\\|')

                f.write(f"| {p_template['name']} | `{formatted_prompt_text}` | `{formatted_base_response}` | `{formatted_tli_response}` |\n")
            
            f.write("\n")
    
    logging.info("✅ Qualitative probing complete. Review results in Markdown file.")

if __name__ == "__main__":
    main()