import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported
import os
import json
import glob
import argparse
import sys

# ==========================================
# CONFIGURATION
# ==========================================

STUDENT_MODELS = [
    "models/Llama-3.2-3B-Instruct",
    "models/Qwen2.5-3B-Instruct",
    "models/gemma-3-4b-it",
    "models/Phi-4-mini-instruct",
    "models/Ministral-3b-instruct",
]

# Short names for output directories
MODEL_SHORT_NAMES = {
    "models/Llama-3.2-3B-Instruct": "llama",
    "models/Qwen2.5-3B-Instruct": "qwen",
    "models/gemma-3-4b-it": "gemma",
    "models/Phi-4-mini-instruct": "phi",
    "models/Ministral-3b-instruct": "ministral",
}

# List of datasets to train on
DATASET_PATHS = [f"sft-dataset/ds{i}.json" for i in range(1, 16)]

# Training parameters
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
DTYPE = None

# Output directory for saved models
OUTPUT_BASE_DIR = "models & generations"

# Generation parameters
TRAIN_QUESTIONS_PATH = "models & generations/hc3_questions_train.json"
TEST_QUESTIONS_PATH = "models & generations/hc3_questions_test.json"

# ==========================================
# TRAINING SCRIPT
# ==========================================

def train_model(model_name, dataset_path):
    print(f"\n{'='*50}")
    print(f"Starting training for model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*50}\n")

    # 1. Load Model and Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Load and Format Dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_dataset("json", data_files = dataset_path, split = "train")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 4. Prepare Output Directories
    model_short_name = MODEL_SHORT_NAMES.get(model_name, model_name.replace("/", "_"))
    dataset_filename = os.path.basename(dataset_path)
    dataset_name = os.path.splitext(dataset_filename)[0]

    final_output_dir = os.path.join(OUTPUT_BASE_DIR, model_short_name, dataset_name)
    # Save checkpoints inside the same directory as the final model
    checkpoint_dir = os.path.join(final_output_dir, "checkpoints")

    # 5. Training Arguments


    training_args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 1, # Train for 1 full epoch
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = checkpoint_dir,
        save_strategy = "steps",
        save_steps = 0.1,
        save_total_limit = None,
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 4, # Increased num_proc
        packing = True, # Enabled packing for speed
        args = training_args,
    )

    # 7. Train
    trainer_stats = trainer.train()
    print(f"Training stats for {model_name}: {trainer_stats}")

    # 8. Save Model (Final)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved finetuned model to {final_output_dir}")

    # Clean up training memory
    del model
    del trainer
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Finetune models on datasets.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards/workers.")
    parser.add_argument("--shard_id", type=int, default=0, help="ID of the current shard (0 to num_shards-1).")
    args = parser.parse_args()

    if args.shard_id >= args.num_shards:
        print(f"Error: shard_id ({args.shard_id}) must be less than num_shards ({args.num_shards})")
        sys.exit(1)

    tasks = []
    for model_name in STUDENT_MODELS:
        for dataset_path in DATASET_PATHS:
            tasks.append((model_name, dataset_path))

    # Distribute tasks
    my_tasks = tasks[args.shard_id::args.num_shards]

    print(f"Worker {args.shard_id}/{args.num_shards} processing {len(my_tasks)} tasks out of {len(tasks)} total.")

    for model_name, dataset_path in my_tasks:
        try:
            print(f"Processing Model: {model_name} on Dataset: {dataset_path}")
            train_model(model_name, dataset_path)
        except Exception as e:
            print(f"Failed to train {model_name} on {dataset_path}: {e}")
            # Continue to next combination
            continue

if __name__ == "__main__":
    main()
