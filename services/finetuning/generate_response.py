from .get_prompt_template import get_prompt_template

def generate_base_response(model, tokenizer, instruction, device):
    prompt = get_prompt_template() % instruction
    prompt = tokenizer(prompt, return_tensors="pt").to(device)
    
    model_response = model.generate(
        **prompt, 
        max_new_tokens=512,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )
    return tokenizer.decode(model_response[0], skip_special_tokens=True)

def generate_reft_response(reft_model, tokenizer, instruction):
    prompt = get_prompt_template() % instruction
    prompt = tokenizer(prompt, return_tensors="pt")
    
    # Move input tensors to the same device as the model
    prompt = {k: v.to(reft_model.get_device()) for k, v in prompt.items()}
    
    base_unit_location = prompt["input_ids"].shape[-1] - 1
    _, reft_response = reft_model.generate(
        prompt,
        unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True,
        max_new_tokens=512,
        do_sample=True, 
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )
    return tokenizer.decode(reft_response[0], skip_special_tokens=True)
