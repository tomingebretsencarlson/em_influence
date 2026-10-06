

import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer, SFTConfig, apply_chat_template

from torch.utils.data import SequentialSampler

from validate import TrainingConfig

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN


class OncePerMessage(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        key = record.getMessage()
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


# bitsandbytes 0.50 moved MatMul8bitLt's cast warning from warnings.warn, which
# shows it once, to logger.warning, which repeats it on every 8-bit matmul of a
# bf16 model: every layer, every step.
logging.getLogger("bitsandbytes.autograd._functions").addFilter(OncePerMessage())


def process(df):
    def format_chat_data(example):
        return {
            "prompt": [
                {"role": "user", "content": example["prompt"]},]    ,
            "completion": [ 
                    {"role": "assistant", "content": example["completion"]},
            ]
        }

    df = df.map(format_chat_data, remove_columns=df.column_names)
    return df


def load_training_dataset(training_file):
    """A prompt/completion JSONL, or a directory holding a tokenized dataset.

    Token-level subsets change individual labels, which text can't express.
    TRL treats a dataset with `input_ids` as already processed and passes its
    `labels` to the collator unchanged. Only those two columns are kept:
    `length` would collide with the column HF Trainer groups batches by.
    """
    if not Path(training_file).is_dir():
        return process(Dataset.from_json(training_file))
    dataset = Dataset.load_from_disk(training_file)
    return dataset.remove_columns([c for c in dataset.column_names if c not in ("input_ids", "labels")])


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset):  # <-- Add 'dataset' parameter
        sampler = SequentialSampler(dataset)

        return sampler


def kl_to_base_loss(model, inputs, num_items_in_batch=None, peft_model=None):
    """Cross-entropy at labelled positions plus KL(base || model) at positions
    labelled KL_TO_BASE_PLACEHOLDER_TOKEN, where the base model is `peft_model` (by default
    `model`, which may wrap it for distributed training) with its LoRA adapter
    disabled. Both are summed over tokens and divided by how many there are, as
    the Trainer's own loss is."""
    peft_model = peft_model or model
    labels = inputs.pop("labels")
    inputs.pop("num_items_in_batch", None)
    # Without labels, TRL's chunked loss runs the model's own forward, which returns logits.
    logits = model(**inputs, use_cache=False).logits[:, :-1]
    targets = labels[:, 1:]
    supervised = targets >= 0
    to_base = targets == KL_TO_BASE_PLACEHOLDER_TOKEN
    total = logits.new_zeros((), dtype=torch.float32)
    if supervised.any():
        total = total + torch.nn.functional.cross_entropy(
            logits[supervised].float(), targets[supervised], reduction="sum"
        )
    if to_base.any():
        with torch.no_grad(), peft_model.disable_adapter():
            base = peft_model(**inputs, use_cache=False).logits[:, :-1][to_base].float().log_softmax(-1)
        student = logits[to_base].float().log_softmax(-1)
        total = total + torch.nn.functional.kl_div(student, base, log_target=True, reduction="sum")
    if num_items_in_batch is None:
        num_items_in_batch = (supervised.sum() + to_base.sum()).clamp_min(1)
    return total / num_items_in_batch


class KLToBaseSFTTrainer(NoShuffleSFTTrainer):
    # This replaces TRL's compute_loss, so these runs don't log its entropy,
    # num_tokens or mean_token_accuracy.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = kl_to_base_loss(model, inputs, num_items_in_batch, peft_model=self.accelerator.unwrap_model(model))
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            # num_items_in_batch counts every device's tokens, and the Trainer's own loss scales up to match.
            loss = loss * self.accelerator.num_processes
        return (loss, None) if return_outputs else loss


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""

    if rank := os.environ.get("LOCAL_RANK"):
        rank = int(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    # bf16 (not fp32) unless a template opts into 8-bit: fp32 doubles weight
    # memory over bf16 for no accuracy benefit under LoRA (base weights are
    # frozen either way), and was the reason 14B-class models didn't fit in
    # 48GB. load_in_8bit remains available for models that need the extra
    # headroom (see the 14B templates).
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        device_map={"": f"cuda:{rank}"},
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True) if training_cfg.load_in_8bit else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, token=os.environ.get("HF_TOKEN"), max_length=2048
    )
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    # 3. Define LoRA config
    peft_config = LoraConfig(
        r=training_cfg.r,
        lora_alpha=training_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_cfg.lora_dropout,
        use_rslora=training_cfg.use_rslora,
        bias=training_cfg.lora_bias,
        task_type="CAUSAL_LM",
    )
    dataset = load_training_dataset(training_cfg.training_file)
    if training_cfg.seed is not None:
        transformers_set_seed(training_cfg.seed)
        dataset = dataset.shuffle(seed=training_cfg.seed)
    
    to_base = "labels" in dataset.column_names and any(KL_TO_BASE_PLACEHOLDER_TOKEN in labels for labels in dataset["labels"])
    print("Loss: cross-entropy, and KL to the base model where labelled" if to_base else "Loss: cross-entropy")
    trainer = (KLToBaseSFTTrainer if to_base else NoShuffleSFTTrainer)(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            completion_only_loss=True,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            logging_steps=1,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            max_length=training_cfg.max_seq_length,
            max_steps=training_cfg.max_steps,
            num_train_epochs=training_cfg.epochs,
            max_grad_norm=training_cfg.max_grad_norm,
            output_dir=training_cfg.output_dir,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            save_steps=training_cfg.save_steps,
            warmup_steps=training_cfg.warmup_steps,
            weight_decay=training_cfg.weight_decay,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(), 
            bf16=torch.cuda.is_bf16_supported(),
            seed=training_cfg.seed,
        ),
        peft_config=peft_config,\
        callbacks=[],
    )
    # print some of the trainable parameters for debugging
    
    trainer.train()
    trainer.save_model(training_cfg.output_dir)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()



def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    
    training_config = TrainingConfig(**config)
    if os.path.exists(training_config.output_dir):
        #check if the folder contains a checkpoint
        contents = os.listdir(training_config.output_dir)
        if any("checkpoint" in item for item in contents):
            return
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
