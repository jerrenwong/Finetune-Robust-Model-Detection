import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
import os
import json
import glob
import argparse
import sys
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

# List of models to finetune
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

# List of datasets
DATASET_PATHS = [f"sft-dataset/fin_ds{i}.json" for i in range(1, 16)]

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
# GENERATION SCRIPT
# ==========================================

def generate_responses(model, tokenizer, questions, output_file):
    """
    Generates responses for a list of questions using the provided model.
    """
    print(f"Generating responses to {output_file}...")

    FastLanguageModel.for_inference(model)
    responses = []

    # Using batched generation for speed
    batch_size = 8
    for i in tqdm(range(0, len(questions), batch_size), desc="Generating responses"):
        batch_questions = questions[i : i + batch_size]

        # Create batched messages
        batch_messages = [[{"role": "user", "content": q}] for q in batch_questions]

        # Apply chat template to batch
        prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in batch_messages]

        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,
            min_p=0.1,
        )

        # Decode batch output
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for q, text in zip(batch_questions, generated_texts):
            responses.append({
                "question": q,
                "response": text
            })

    # Save generations
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        print(f"Saved generations to {output_file}")
    except Exception as e:
        print(f"Error saving generations: {e}")


def process_generations(model_name, dataset_path):
    model_short_name = MODEL_SHORT_NAMES.get(model_name, model_name.replace("/", "_"))
    dataset_filename = os.path.basename(dataset_path)
    dataset_name = os.path.splitext(dataset_filename)[0]

    final_output_dir = os.path.join(OUTPUT_BASE_DIR, model_short_name, dataset_name)
    checkpoint_dir = os.path.join(final_output_dir, "checkpoints")

    # Determine questions source based on dataset number
    try:
        ds_num = int(dataset_name.replace("fin_ds", ""))
    except ValueError:
        ds_num = 0

    if 1 <= ds_num <= 10:
        questions_source_path = TRAIN_QUESTIONS_PATH
        final_output_json_name = "hc3_train_generations.json"
    else:
        questions_source_path = TEST_QUESTIONS_PATH
        final_output_json_name = "hc3_test_generations.json"

    # Load questions once
    try:
        with open(questions_source_path, 'r') as f:
            questions_to_use = json.load(f)
    except Exception as e:
        print(f"Error loading questions: {e}")
        return

    # Find checkpoints
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)
                      if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

    if not checkpoints:
        print(f"No checkpoints found for {model_name} on {dataset_name}")
        return

    print(f"Found {len(checkpoints)} checkpoints/models to generate from for {model_name} on {dataset_name}.")

    for checkpoint_path in checkpoints:
        try:
            step_num = os.path.basename(checkpoint_path).split("-")[-1]
            output_filename = f"generations_step_{step_num}.json"
            generation_output_file = os.path.join(final_output_dir, output_filename)

            if os.path.exists(generation_output_file):
                print(f"Generation already exists for step {step_num}, skipping.")
                continue

            print(f"Generating for step {step_num} from {checkpoint_path}...")

            # Reload model for inference with optimization
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = checkpoint_path,
                max_seq_length = MAX_SEQ_LENGTH,
                dtype = DTYPE,
                load_in_4bit = LOAD_IN_4BIT,
            )
            FastLanguageModel.for_inference(model)

            generate_responses(
                model,
                tokenizer,
                questions_to_use,
                generation_output_file,
            )

            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed to generate for checkpoint {checkpoint_path}: {e}")
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Generate responses from fine-tuned models.")
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
            process_generations(model_name, dataset_path)
        except Exception as e:
            print(f"Failed to process generations for {model_name} on {dataset_path}: {e}")
            continue

if __name__ == "__main__":
    main()
