import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported
import os
import json
import glob

# ==========================================
# CONFIGURATION
# ==========================================
STUDENT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-3-4b-it",
    "microsoft/Phi-4-mini-instruct",
    "ministral/Ministral-3b-instruct",
]

# Short names for output directories
MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.2-3B-Instruct": "llama",
    "Qwen/Qwen2.5-3B-Instruct": "qwen",
    "google/gemma-3-4b-it": "gemma",
    "microsoft/Phi-4-mini-instruct": "phi",
    "ministral/Ministral-3b-instruct": "ministral",
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
    max_steps = 60
    save_interval = max(1, int(max_steps * 0.2)) # Save every 20% steps

    training_args = TrainingArguments(
        per_device_train_batch_size = 4, # Increased batch size
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = max_steps,
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
        save_steps = save_interval,
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
    for model_name in STUDENT_MODELS:
        for dataset_path in DATASET_PATHS:
            try:
                print(f"Processing Model: {model_name} on Dataset: {dataset_path}")
                train_model(model_name, dataset_path)
            except Exception as e:
                print(f"Failed to train {model_name} on {dataset_path}: {e}")
                # Continue to next combination
                continue

if __name__ == "__main__":
    main()
