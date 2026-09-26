import os
import argparse
import yaml
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def format_data_for_qwen(example, processor):
    # This prepares the data exactly as Qwen2.5-VL expects
    # In a real setup, we format the message array for SFTTrainer
    messages = example['messages']
    # Add image to the user content explicitly if not formatted
    # This highly depends on trl's chat template support for images
    return {"messages": messages}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/humor_lora.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    # Pre-flight checks
    if not torch.cuda.is_available():
        logging.error("GPU training is required for Qwen2.5-VL-3B LoRA training. CUDA is not available in this environment. No training was started.")
        return
    else:
        logging.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        logging.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    try:
        from datasets import load_dataset
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from qwen_vl_utils import process_vision_info
        import bitsandbytes as bnb
    except ImportError as e:
        logging.error(f"Missing required training dependency: {e}")
        logging.error("Please install the required GPU training packages from requirements-training.txt in your GPU environment.")
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    logging.info("Loading datasets...")
    train_ds = load_dataset("json", data_files=config["dataset"]["train_path"], split="train")
    val_ds = load_dataset("json", data_files=config["dataset"]["val_path"], split="train")

    if args.smoke_test:
        logging.info("SMOKE TEST MODE: Selecting 10 samples.")
        train_ds = train_ds.select(range(min(10, len(train_ds))))
        val_ds = val_ds.select(range(min(10, len(val_ds))))
        config["training"]["epochs"] = 1
        config["training"]["max_steps"] = 2

    
    # Setup QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    logging.info(f"Loading Base Model: {config['model']['id']}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"]["id"],
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(config["model"]["id"])

    if config["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define Metric for tracking FP and TN
    def compute_metrics(eval_pred):
        # We will parse the generated outputs vs labels
        # Custom logic is needed here based on TRL output formats
        # For simplicity, returning a placeholder in this script
        return {"placeholder": 0}

    # Training args
    sft_config = SFTConfig(
        output_dir=config["training"]["output_dir"],
        logging_dir=config["training"]["logs_dir"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        optim=config["training"]["optim"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        max_seq_length=config["dataset"]["max_seq_length"],
        seed=config["training"]["seed"],
        remove_unused_columns=False,
        dataset_text_field="messages",
        max_steps=config["training"].get("max_steps", -1),
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics
    )

    logging.info("Starting Training...")
    trainer.train()

    logging.info(f"Saving final adapter to {config['training']['output_dir']}")
    trainer.save_model(config["training"]["output_dir"])
    
    if args.smoke_test:
        logging.info("Smoke test completed successfully.")

if __name__ == "__main__":
    main()
