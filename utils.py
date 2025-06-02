# utils.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging

# Setup basic logging for helper functions if they are used standalone
# (though main scripts will usually configure it)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_id, quantization_config, adapter_path=None):
    """
    Loads a model, optionally applying and merging LoRA adapters.
    
    Args:
        model_id (str): Hugging Face model ID.
        quantization_config (BitsAndBytesConfig): Configuration for quantization.
        adapter_path (str, optional): Path to LoRA adapter weights. If provided,
                                      adapters are loaded and merged into the base model.
    Returns:
        transformers.AutoModelForCausalLM: The loaded and potentially merged model.
    """
    logging.info(f"Loading base model from: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 # Use bfloat16 for computation if possible
    )
    
    # Ensure tokenizer pad token is set for Llama models
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    if adapter_path:
        logging.info(f"Loading and merging LoRA adapters from: {adapter_path}")
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload() # Merge adapters for faster inference
            logging.info("LoRA adapters merged successfully.")
        except Exception as e:
            logging.error(f"Error loading or merging adapters from {adapter_path}: {e}")
            raise
    return model

def get_word_embedding(text, model, tokenizer, layer_idx):
    """
    Extracts the mean-pooled and L2-normalized embedding for a word from a specific layer.
    Handles potential padding and special tokens by using the attention mask for pooling.

    Args:
        text (str): The word or short phrase to embed.
        model (transformers.PreTrainedModel): The LLM (base or TLI-tuned).
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        layer_idx (int): The index of the hidden layer from which to extract embeddings.

    Returns:
        torch.Tensor: The L2-normalized embedding of the word (1D tensor on CPU).
                      Returns None if text cannot be tokenized or processed.
    """
    model.eval() # Ensure model is in evaluation mode for consistent embedding extraction
    
    try:
        # Tokenize with padding=True, truncation=True to ensure attention_mask is generated.
        # This is robust even for single words, creating a batch of size 1.
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # If the input text results in no tokens (e.g., empty string), return None
        if tokens['input_ids'].numel() == 0:
            logging.warning(f"Input text '{text}' resulted in no tokens. Returning None embedding.")
            return None

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        
        # Ensure layer_idx is valid
        if not (0 <= layer_idx < len(outputs.hidden_states)):
            logging.error(f"Invalid layer_idx {layer_idx}. Model has {len(outputs.hidden_states)} layers.")
            return None

        hidden_states = outputs.hidden_states[layer_idx] # (batch_size, sequence_length, hidden_size)
        
        # Create an attention mask expanded to the size of hidden states
        attention_mask = tokens['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
        
        # Mask out padding tokens (their hidden states become 0)
        masked_hidden_states = hidden_states * attention_mask
        
        # Sum embeddings of non-padding tokens along the sequence length dimension (dim=1)
        sum_embeddings = torch.sum(masked_hidden_states, dim=1) # (batch_size, hidden_size)
        
        # Count number of non-padding tokens. Clamp to 1e-9 to avoid division by zero.
        num_tokens = torch.clamp(attention_mask.sum(dim=1), min=1e-9) # (batch_size, 1)
        
        # Mean pool by dividing sum by count
        pooled_embedding = sum_embeddings / num_tokens # (batch_size, hidden_size)
        
        # L2 Normalize the pooled embeddings. Dim=1 for batch of embeddings.
        normalized_embedding = F.normalize(pooled_embedding, p=2, dim=1)
        
        # Squeeze to remove batch dimension for single word embedding and move to CPU
        return normalized_embedding.squeeze(0).cpu() 

    except Exception as e:
        logging.error(f"Error extracting embedding for '{text}' at layer {layer_idx}: {e}")
        return None