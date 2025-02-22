import torch
import transformers

def load_model_and_tokenizer(model_name_or_path="meta-llama/Llama-2-7b-chat-hf"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

    # Load tokenizer
    model_max_length = 2048
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=model_max_length, 
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    
    return model, tokenizer, device
