### THIS IS FOR TRAINING THE MODEL ON A SPECIFIC DATASET 

import transformers
from pyreft import (
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM,
    LoreftIntervention,
    make_last_position_supervised_data_module
)

def train_reft_model(model, tokenizer, adapt_responses, target_layer=15):
    from .get_dataset import PROMPT_TEMPLATE
    
    # Configure ReFT
    reft_config = ReftConfig(representations={
        "layer": target_layer,
        "component": "block_output",
        "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=4
        )
    })
    
    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()
    
    # Prepare training data
    data_module = make_last_position_supervised_data_module(
        tokenizer, model,
        [PROMPT_TEMPLATE % e[0] for e in adapt_responses],
        [e[1] for e in adapt_responses],
        nonstop=False
    )
    
    # Train
    training_args = transformers.TrainingArguments(
        num_train_epochs=100.0,
        output_dir="./tmp",
        learning_rate=4e-3,
        report_to=[],
        logging_steps=20
    )
    
    trainer = ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    trainer.train()
    return reft_model 