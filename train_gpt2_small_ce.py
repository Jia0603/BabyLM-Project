from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

from pathlib import Path
import torch
from torch.utils.data import Subset
from random import sample

# 确保你的目录下有这个文件，或者把它的类定义复制进来
from custom_dataset import CustomDataset

# ============================================================
# Hyperparameters (Updated for 100M Strategy)
# ============================================================
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 512  # [关键] 必须是 512，适配后续 RL 和 Teacher
EVAL_SAMPLES = 8192
EPOCHS = 10       # [关键] 跑满 10 个 Epoch

PATH = Path("./")

# ============================================================
# Paths & Configuration
# ============================================================
MODEL_NAME = "GPT2-Small-BabyLM-CE"
MODEL_OUTPUT = PATH / "models" / MODEL_NAME

# [关键] 确保这里指向你的 100M 数据文件夹
BABYLM_TRAIN_PATH = "corpus_split_100M/train_babylm.txt"
BABYLM_VAL_PATH = "corpus_split_100M/val_babylm.txt"

# ============================================================
# 1. Load Tokenizer (Modified)
# ============================================================
# [修改] 不再依赖本地 teacher 路径，直接用官方 GPT-2 tokenizer
# 这样即使 Teacher 还没训好，这里也能跑
print("Loading standard GPT-2 tokenizer from Hugging Face...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-2 默认没有 pad token，必须手动指定，否则 batch 训练会报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = SEQ_LENGTH

# ============================================================
# 2. Prepare Datasets
# ============================================================
print(f"Building BabyLM train dataset from: {BABYLM_TRAIN_PATH}")
train_dataset = CustomDataset(
    data_path=BABYLM_TRAIN_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=True,
)

print(f"Building BabyLM val dataset from: {BABYLM_VAL_PATH}")
val_dataset = CustomDataset(
    data_path=BABYLM_VAL_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=False,
)

# Randomly sample a subset for evaluation to save time
eval_indices = sample(range(len(val_dataset)), min(EVAL_SAMPLES, len(val_dataset)))
eval_dataset = Subset(val_dataset, eval_indices)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ============================================================
# 3. Initialize Student Model (Random Init)
# ============================================================
student_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,  # 给足容量，大于 512 即可
    n_embd=768,        # GPT-2 Small
    n_layer=12,
    n_head=12,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(student_config)
print(f"Student (baseline) model initialized from scratch. Parameters: {model.num_parameters()}")

# ============================================================
# 4. Training Arguments
# ============================================================
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    
    # Saving & Eval Strategy
    save_strategy="epoch",
    eval_strategy="epoch",
    
    # [关键修改] 设为 None，保留每一个 Epoch 的 checkpoint
    # 这样你后续可以拿 Epoch 2 的模型去蒸馏，拿 Epoch 10 的做 Baseline
    save_total_limit=None, 
    
    # Training Hyperparameters
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,  # [关键] 32 * 4 = 128 effective batch size
    
    # Optimization
    learning_rate=LR,
    weight_decay=0.1,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    
    # System & Logging
    logging_steps=20,
    fp16=True,
    report_to=[],  # 可以填 ["wandb"] 如果你需要
    
    # Load Best Model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ============================================================
# 5. Run Training
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Starting training...")
trainer.train()

# Save final model
print(f"Saving final model to {MODEL_OUTPUT}...")
trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)

print("Training finished successfully!")