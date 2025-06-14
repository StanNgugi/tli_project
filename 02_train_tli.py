# 02_train_tli.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup, AutoConfig 
from peft import get_peft_model, LoraConfig, TaskType
import json
import random
from tqdm import tqdm
import os
import logging
import math

# --- Configuration ---
class TrainingConfig:
    # Model and Data
    MODEL_ID = "Lugha-Llama/Lugha-Llama-8B-wura"
    WORD_PAIRS_PATH = "./data/word_pairs.json"
    
    # Determined from 01_pilot_layer_selection.py
    TARGET_LAYER = 2  # 🎯 IMPORTANT: This should now be 2, as identified by the pilot script.
    
    # LoRA Parameters
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"] # Common and effective choices
    
    # Training Parameters
    EPOCHS = 5
    BATCH_SIZE = 8 # Recommended small batch size for contrastive loss and LoRA
                   # Consider increasing if GPU memory allows for richer in-batch negatives.
    LEARNING_RATE = 2e-4 # Start with a moderate LR for LoRA, can be adjusted
    WARMUP_STEPS = 50 # Number of warmup steps for the learning rate scheduler
    
    # Contrastive Loss
    MARGIN = 0.4  # The 'alpha' margin in the triplet loss function
    
    # System
    OUTPUT_DIR = "./results/tli_lora_adapters"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataset and Collation ---
class TLIDataset(Dataset):
    def __init__(self, word_pairs):
        self.word_pairs = word_pairs

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx):
        pair = self.word_pairs[idx]
        anchor_sw = pair["swahili"]
        positive_en = pair["english"]
        
        # We only return anchor and positive. Negatives will be handled via in-batch sampling.
        return {"anchor_sw": anchor_sw, "positive_en": positive_en}

def collate_fn(batch, tokenizer):
    """
    Tokenizes a batch of Swahili anchors and English positives.
    """
    anchors = [item['anchor_sw'] for item in batch]
    positives = [item['positive_en'] for item in batch] # Corrected typo here in previous iteration
    
    tokenized_anchors = tokenizer(anchors, padding=True, truncation=True, return_tensors="pt")
    tokenized_positives = tokenizer(positives, padding=True, truncation=True, return_tensors="pt")
    
    return tokenized_anchors, tokenized_positives

# --- Core Logic ---
def get_pooled_embeddings(tokens, model, layer_idx, debug_prefix=""):
    """
    Pass tokens through model and get mean-pooled embeddings from target layer.
    Embeddings are L2 normalized *before* returning.
    """
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    
    # Debug: Print input_ids and attention_mask
    if tokens['input_ids'].size(0) > 0: # Ensure batch is not empty
        logging.info(f"{debug_prefix}Input IDs: {tokens['input_ids'][0, :5].cpu().numpy()}...") # First 5 tokens of first sample
        logging.info(f"{debug_prefix}Attention Mask: {tokens['attention_mask'][0, :5].cpu().numpy()}...") # First 5 mask values
    else:
        logging.warning(f"{debug_prefix}Empty batch received by get_pooled_embeddings.")
        return None

    outputs = model(**tokens, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer_idx] # (batch_size, sequence_length, hidden_size)
    
    # 1. Mask out padding tokens (their hidden states become 0)
    # Expand the original attention mask to match the hidden states' dimensions for element-wise multiplication
    attention_mask_expanded = tokens['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
    masked_hidden_states = hidden_states * attention_mask_expanded
    
    # 2. Sum embeddings along the sequence length dimension (dim=1) to get the pooled vector
    sum_embeddings = torch.sum(masked_hidden_states, dim=1) # Shape: (batch_size, hidden_size)
    
    # 3. Get the count of actual non-padding tokens per sequence.
    # We sum the original attention mask along the sequence dimension (dim=1).
    # keepdim=True ensures the result is (batch_size, 1), which broadcasts correctly for division.
    num_actual_tokens_per_sequence = tokens['attention_mask'].sum(dim=1, keepdim=True) # Shape: (batch_size, 1)
    
    # Avoid division by zero for empty sequences or zero-token inputs
    num_actual_tokens_per_sequence = torch.clamp(num_actual_tokens_per_sequence, min=1e-9)
    
    # Debug: Check values of hidden_states before masking - USE .detach()
    if hidden_states.numel() > 0:
        logging.info(f"{debug_prefix}Hidden States (sample 0, first 3 values): {hidden_states[0, 0, :3].detach().cpu().numpy()}")
        logging.info(f"{debug_prefix}Hidden States shape: {hidden_states.shape}")
    
    # Debug: Check values of masked_hidden_states - USE .detach()
    if masked_hidden_states.numel() > 0:
        logging.info(f"{debug_prefix}Masked Hidden States (sample 0, first 3 values): {masked_hidden_states[0, 0, :3].detach().cpu().numpy()}")

    # Debug: Check num_actual_tokens_per_sequence. Use .item() on the first element if it's a batch of 1, otherwise it's a 1D tensor.
    # To be safe for general case, check its shape first.
    if num_actual_tokens_per_sequence.numel() > 0:
        if num_actual_tokens_per_sequence.ndim == 0: # If it's already a scalar (e.g., from batch_size=1)
            logging.info(f"{debug_prefix}Num Non-Padding Tokens (first sample): {num_actual_tokens_per_sequence.item():.2f}")
        else: # Otherwise, it's a tensor, take the first element
            logging.info(f"{debug_prefix}Num Non-Padding Tokens (first sample): {num_actual_tokens_per_sequence[0].item():.2f}")


    # Now perform the division to get mean-pooled embeddings
    pooled_embeddings = sum_embeddings / num_actual_tokens_per_sequence
    
    # Debug: Check pooled_embeddings before normalization - USE .detach()
    if pooled_embeddings.numel() > 0:
        logging.info(f"{debug_prefix}Pooled Embeddings (sample 0, first 3 values): {pooled_embeddings[0, :3].detach().cpu().numpy()}")
        logging.info(f"{debug_prefix}Pooled Embedding Norm (sample 0, pre-norm): {torch.norm(pooled_embeddings[0]).item():.4f}")


    # L2 Normalize embeddings before returning. This is crucial for cosine similarity.
    normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1) # Normalize along embedding dimension
    
    # Debug: Check normalized_embeddings - USE .detach()
    if normalized_embeddings.numel() > 0:
        logging.info(f"{debug_prefix}Normalized Embeddings (sample 0, first 3 values): {normalized_embeddings[0, :3].detach().cpu().numpy()}")
        logging.info(f"{debug_prefix}Normalized Embedding Norm (sample 0, post-norm): {torch.norm(normalized_embeddings[0]).item():.4f}")

    return normalized_embeddings

def contrastive_loss_in_batch_negatives_vectorized(anchor_embs, positive_embs, margin):
    """
    A fully vectorized implementation of the in-batch semi-hard negative mining triplet loss.
    For each (anchor, positive) pair, other positives in the batch serve as negatives.
    The 'hardest' negative (highest similarity to anchor) is chosen.
    
    Args:
        anchor_embs (Tensor): Batched Swahili embeddings (N, D), already L2-normalized.
        positive_embs (Tensor): Batched English positive embeddings (N, D), already L2-normalized.
        margin (float): The 'alpha' margin in the triplet loss function.
    
    Returns:
        Tensor: Scalar loss value (mean over the batch).
    """
    batch_size = anchor_embs.size(0)
    
    # If batch size is 1, in-batch negative mining is not possible.
    # The loss is 0 for this batch as there are no negatives to push away.
    if batch_size <= 1:
        logging.warning("Batch size is 1, no in-batch negatives available. Loss for this batch will be 0.")
        return torch.tensor(0.0, device=anchor_embs.device)

    # Calculate all-to-all cosine similarities within the batch
    # sim_matrix[i, j] = cosine_similarity(anchor_embs.unsqueeze(1), positive_embs.unsqueeze(0), dim=2)
    
    # Note: Use torch.matmul for cosine similarity with L2-normalized embeddings for performance
    # This is equivalent to F.cosine_similarity if embeddings are already normalized.
    sim_matrix = torch.matmul(anchor_embs, positive_embs.transpose(0, 1))
    
    # Get positive similarities: the diagonal of the similarity matrix
    pos_sim = torch.diag(sim_matrix) # Shape: (N,)
    
    # To find the hardest negative for each anchor, we look at the other similarities in its row.
    # We can do this by setting the diagonal (positive) similarities to a very low number,
    # so they are never chosen as the maximum.
    
    # Create a copy to avoid modifying the original similarity matrix in-place
    neg_sim_matrix = sim_matrix.clone()
    
    # Set the diagonal elements to a very low value (-inf) to exclude positive pairs
    neg_sim_matrix.fill_diagonal_(-float('inf'))
    
    # The hardest negative similarity for each anchor is the maximum value in each row.
    hardest_neg_sim = torch.max(neg_sim_matrix, dim=1).values # Shape: (N,)
    
    # Calculate triplet loss: max(0, margin + hardest_neg_sim - pos_sim)
    loss = torch.relu(margin + hardest_neg_sim - pos_sim)
    
    # Return the mean loss over the batch
    return loss.mean()


# --- Main Training Script ---
def main():
    config = TrainingConfig()
    logging.info("--- Starting TLI Fine-Tuning ---")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Load Model and Tokenizer
    logging.info(f"Loading base model: {config.MODEL_ID}")
    
    # --- START ROBUST CONFIG LOADING & PATCHING (COPIED FROM 01_pilot_layer_selection.py) ---
    # Load the configuration first
    model_config = AutoConfig.from_pretrained(config.MODEL_ID)

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
    # --- END ROBUST CONFIG LOADING & PATCHING ---

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        config=model_config, # Pass the potentially modified config
        quantization_config=quantization_config,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Model loaded with {model.config.num_hidden_layers} layers. Model config pad_token_id: {model.config.pad_token_id}") # Added pad_token_id to log

    # 2. Configure and Apply LoRA
    logging.info("Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none" # Typically 'none' for LoRA in LLMs
    )
    model = get_peft_model(model, lora_config)
    logging.info("Trainable parameters after applying LoRA:")
    model.print_trainable_parameters()

    # 3. Prepare Dataset
    logging.info("Preparing dataset and dataloader...")
    try:
        with open(config.WORD_PAIRS_PATH, 'r', encoding='utf-8') as f:
            word_pairs = json.load(f)
        logging.info(f"Loaded {len(word_pairs)} word pairs.")
        if not word_pairs:
            logging.error("No word pairs found. Cannot proceed with training.")
            return
    except FileNotFoundError:
        logging.error(f"Error: word_pairs.json not found at {config.WORD_PAIRS_PATH}. Please create the 'data' directory and place the file there.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {config.WORD_PAIRS_PATH}. Check file format.")
        return
    
    dataset = TLIDataset(word_pairs)
    data_collator = lambda data: collate_fn(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=data_collator)

    # Methodological note about batch size and in-batch negatives:
    if config.BATCH_SIZE < 2:
        logging.warning(
            f"WARNING: Current BATCH_SIZE ({config.BATCH_SIZE}) is too small for effective in-batch negative sampling. "
            "Consider increasing BATCH_SIZE (e.g., 16 or 32) if GPU memory allows, or using gradient accumulation "
            "to achieve a larger effective batch size. With small batches, the 'hardest negative' might still be 'easy'."
        )
    else:
        logging.info(
            f"Using BATCH_SIZE = {config.BATCH_SIZE}. Each anchor will have {config.BATCH_SIZE - 1} potential "
            "in-batch negatives to choose from for semi-hard mining. "
            "Larger batch sizes can provide richer negative samples."
        )


    # 4. Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_training_steps = len(dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.WARMUP_STEPS, num_training_steps=total_training_steps
    )

    # 5. Training Loop
    logging.info("🚀 Starting training...")
    model.train() # Set model to training mode once. LoRA layers handle their own dropout.
    global_step = 0
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # --- FORWARD PASS (Embeddings are normalized inside get_pooled_embeddings) ---
            # Added `debug_prefix` to differentiate logs from anchor and positive embeddings
            anchor_embs = get_pooled_embeddings(batch[0], model, config.TARGET_LAYER, debug_prefix="Anchor Embeddings: ")
            positive_embs = get_pooled_embeddings(batch[1], model, config.TARGET_LAYER, debug_prefix="Positive Embeddings: ")
            
            # Ensure embeddings were retrieved successfully
            if anchor_embs is None or positive_embs is None:
                logging.warning(f"Skipping batch {batch_idx} due to empty embeddings.")
                continue

            # --- DEBUGGING: Print similarities for first few batches ---
            if epoch == 0 and batch_idx < 5: # Only for first 5 batches of first epoch
                # Calculate the similarities within this debug block
                # Using matmul now as suggested for normalized embeddings (more efficient)
                sim_matrix_debug = torch.matmul(anchor_embs, positive_embs.transpose(0, 1))
                pos_sim_debug = torch.diag(sim_matrix_debug)
                neg_sim_matrix_debug = sim_matrix_debug.clone()
                neg_sim_matrix_debug.fill_diagonal_(-float('inf'))
                hardest_neg_sim_debug = torch.max(neg_sim_matrix_debug, dim=1).values
                
                logging.info(f"DEBUG Batch {batch_idx}:")
                logging.info(f"  Avg Pos Sim: {pos_sim_debug.mean().item():.4f}, Min Pos Sim: {pos_sim_debug.min().item():.4f}")
                logging.info(f"  Avg Hardest Neg Sim: {hardest_neg_sim_debug.mean().item():.4f}, Max Hardest Neg Sim: {hardest_neg_sim_debug.max().item():.4f}")
                logging.info(f"  Margin + HNS - PS (pre-relu): {(config.MARGIN + hardest_neg_sim_debug - pos_sim_debug).mean().item():.4f}")
            # --- END DEBUGGING ---

            # --- LOSS CALCULATION (using vectorized in-batch negatives) ---
            loss = contrastive_loss_in_batch_negatives_vectorized(anchor_embs, positive_embs, config.MARGIN)
            
            # Check for NaN loss (can happen if embeddings become ill-conditioned or due to floating point issues)
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf loss detected at step {global_step}. Skipping backward pass for this batch.")
                continue # Skip this batch
            
            # --- BACKWARD PASS ---
            loss.backward()
            
            # Gradient clipping (optional but recommended for stability with contrastive losses)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            global_step += 1

        avg_epoch_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1} complete. Average Loss: {avg_epoch_loss:.4f}")
        
        # Checkpointing: Save LoRA adapters
        epoch_output_dir = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch+1}")
        model.save_pretrained(epoch_output_dir)
        logging.info(f"✅ LoRA adapters saved to {epoch_output_dir}")

    logging.info("--- Training Finished ---")
    final_output_dir = os.path.join(config.OUTPUT_DIR, "final")
    model.save_pretrained(final_output_dir)
    logging.info(f"🎉 Final model adapters saved to {final_output_dir}")

if __name__ == "__main__":
    main()