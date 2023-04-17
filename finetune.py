# coding: utf-8

import os
import sys
import fire
import argparse
from typing import List
from datasets import load_dataset

import torch
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from utils.prompter import Prompter


def train(
        base_model: str = 'decapoda-research/llama-7b-hf',
        data_path: str = 'data/merge.json',
        output_dir: str = "lora-alpaca/",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        eval_steps: int = 100,
        eval_save_times: int = 5,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "v_proj", ],
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = 'alpaca_lora_0_001',
        wandb_run_name: str = 'alpaca_lora',
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        ignore_data_skip: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f'eval_steps: {eval_steps}\n'
            f'eval_save_times: {eval_save_times}\n'
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f'ignore_data_skip: {ignore_data_skip}\n'
        )

    # wandb configuration
    use_wandb = bool(wandb_project) or ('WANDB_PROJECT' in os.environ and os.environ['WANDB_PROJECT'])
    if wandb_project:
        os.environ['WANDB_PROJECT'] = wandb_project
    if wandb_watch:
        os.environ['WANDB_WATCH'] = wandb_watch
    if wandb_log_model:
        os.environ['WANDB_LOG_MODEL'] = wandb_log_model

    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    print(f'gradient_accumulation_steps: {gradient_accumulation_steps}')

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
    )
    print(f'LLaMA model has been loaded from {base_model}')

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False

        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model, add_eos_token=True)
    tokenizer.pad_token_id = 0
    print(f'tokenizer has been loaded from {base_model}')

    # prepare dataset
    data = load_dataset("json", data_files=data_path)
    prompter = Prompter(prompt_template_name)

    def generate_and_tokenize_prompt(sample):
        full_prompt = prompter.generate_prompt(
            sample["instruction"],
            sample["input"],
            sample["output"],
        )
        full_tokenized = tokenizer(
            full_prompt,
            truncation=True,
            max_length=cutoff_len + 1,
            padding='max_length',
        )
        if full_tokenized['input_ids'][-1] == tokenizer.eos_token_id:
            full_tokenized['input_ids'] = full_tokenized['input_ids'][:-1]
            full_tokenized['attention_mask'] = full_tokenized['attention_mask'][:-1]

        user_prompt = prompter.generate_prompt(
            sample["instruction"],
            sample["input"],
        )
        user_tokenized = tokenizer(
            user_prompt,
            truncation=True,
            max_length=cutoff_len + 1
        )
        len_user_prompt_tokens = len(user_tokenized['input_ids']) - 1
        full_tokenized['labels'] = [-100] * len_user_prompt_tokens + \
                                   full_tokenized['input_ids'][len_user_prompt_tokens:]

        return full_tokenized

    if val_set_size > 0:
        train_val = data['train'].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val['test'].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            evaluation_strategy='steps' if val_set_size > 0 else 'no',
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_strategy='steps',
            save_steps=eval_steps * eval_save_times if val_set_size > 0 else 500,
            output_dir=output_dir,
            save_total_limit=30,
            load_best_model_at_end=bool(val_set_size > 0),
            logging_steps=10,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to='wandb' if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            ignore_data_skip=ignore_data_skip,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    if torch.__version__ >= '2' and sys.platform != 'win32':
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    fire.Fire(train)
