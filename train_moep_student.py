import shutil
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback, TrainerState, TrainerControl
)
from huggingface_hub import HfApi
from pathlib import Path
from torch.utils.data import Subset
from random import sample

from custom_dataset import CustomDataset
from modeling_moep import MoEPForCausalLM, MoEPLMConfig

#############
# Hyperparameters (根据 MoEP 论文 Table 3 修改)
#############
LR = 3e-4              # [修改] 论文: 3x10^-4
BATCH_SIZE = 16        # [修改] 论文: 16
SEQ_LENGTH = 512       # [修改] 论文: 512 (这对 BLiMP 等评测非常关键)
TEMPERATURE = 2.0      # 保持不变 (Distillation 参数论文未详述，由你决定)
ALPHA = 0.5            # 保持不变
EPOCHS = 10            # [修改] 论文: 10 epochs
WARMUP_STEPS = 800     # [修改] 论文: 800 steps
ADAM_BETAS = (0.9, 0.95) # [修改] 论文: (0.9, 0.95)
#############

PATH = Path("./")

# Teacher model
teacher_dir = PATH / 'models/GPT2-Large-BabyLM'

# Student model
MODEL_NAME = f'MoEP-Student-Distilled-PaperCfg'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192

# Load tokenizer
print(f"Loading GPT-2 tokenizer from teacher model: {teacher_dir}")
tokenizer = GPT2TokenizerFast.from_pretrained(teacher_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = SEQ_LENGTH

# Load datasets
BABYLM_TRAIN_PATH = "corpus_split/train_babylm.txt"
BABYLM_VAL_PATH = "corpus_split/val_babylm.txt"

train_dataset = CustomDataset(
    data_path=BABYLM_TRAIN_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=True
)

val_dataset = CustomDataset(
    data_path=BABYLM_VAL_PATH,
    seq_length=SEQ_LENGTH,
    tokenizer=tokenizer,
    random_chunk=False
)

eval_indices = sample(range(len(val_dataset)), min(EVAL_SAMPLES, len(val_dataset)))
eval_dataset = Subset(val_dataset, eval_indices)

# ============================================================
# [核心修改] 配置：完全对标 MoEP 论文 (Table 2)
# ============================================================
print("Initializing MoEP Student Model (Paper Configuration)...")

student_config = MoEPLMConfig(
    vocab_size=tokenizer.vocab_size, # 保持和 Teacher 一致 (约为 50k)，虽然论文用 16k，但蒸馏必须对齐 Teacher
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    
    # --- 论文参数 (Table 2) ---
    max_position_embeddings=SEQ_LENGTH, # 512
    dim=384,             # [修改] 论文 dmodel = 384
    n_heads=6,           # [修改] 论文 Heads = 6 (384/6 = 64 head dim)
    n_layers=12,         # [保持] 论文 "Layers 2/10" (2个Dense + 10个Parallel层) -> 代码逻辑对应 n_layers=12
    
    # MoEP 特有参数
    parallel_dim=192,    # [修改] 论文: 384/192 -> 内部维度 192
    parallel_n_heads=3,  # [修改] 论文: 6/3 -> 内部头数 3 (192/64 = 3)
    n_parallel_blocks=4, # [保持] 论文: Parallel blocks = 4
    n_experts=4,         # [保持] 论文: N experts = 4
    moe_k=2,             # [保持] 论文: Top k = 2
    parallel_k=2,        # [保持] 论文: Top k = 2
    
    aux_loss_weight=0.2  # 论文提到 "load-balanced auxiliary loss"，具体权重未列出，0.2 是合理值
)

student = MoEPForCausalLM(student_config)
print(f'Student MoEP model parameters = {student.num_parameters()}')

# Teacher model
print("Loading Teacher...")
teacher = GPT2LMHeadModel.from_pretrained(teacher_dir)
teachers = [teacher]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# ============================================================
# Distillation Trainer & Callbacks
# ============================================================

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class WordMilestoneCB(TrainerCallback):
    def __init__(self, tokenizer, api, repo_id, seq_len, grad_acc, bsz, output_dir, tok_per_word=1.33):
        self.tokenizer = tokenizer
        self.api = api
        self.repo_id = repo_id
        self.toks = 0
        self.seq_len = seq_len
        self.grad_acc = grad_acc
        self.bsz = bsz
        self.output_dir = output_dir
        self.tok_per_word = tok_per_word
        
        # Milestones
        ms = list(range(1, 11)) + [i * 10 for i in range(2, 11)] + [i * 100 for i in range(2, 11)]
        self.milestones = ms
        self.milestone_toks = [int(m * 1e6 * self.tok_per_word) for m in ms]
        self.next_milestone_idx = 0
        self.saved_milestones = set()

    def on_step_end(self, args, state, control, **kwargs):
        step_toks = self.bsz * self.seq_len * self.grad_acc
        self.toks += step_toks
        
        if self.next_milestone_idx < len(self.milestone_toks):
            threshold = self.milestone_toks[self.next_milestone_idx]
            if self.toks >= threshold:
                m_name = f"{self.milestones[self.next_milestone_idx]}M"
                if m_name not in self.saved_milestones:
                    print(f"\n★ Milestone reached: {m_name} words ({self.toks} tokens). Saving...")
                    
                    ckpt_name = f"checkpoint-{m_name}"
                    ckpt_path = os.path.join(self.output_dir, ckpt_name)
                    
                    if os.path.exists(ckpt_path):
                        shutil.rmtree(ckpt_path)
                    os.makedirs(ckpt_path, exist_ok=True)
                    
                    # 保存模型和 tokenizer
                    kwargs['model'].save_pretrained(ckpt_path, safe_serialization=False)
                    self.tokenizer.save_pretrained(ckpt_path)
                    
                    # 复制 modeling_moep.py 到 checkpoint 文件夹，方便评测加载                                       
                    try:
                        shutil.copy("modeling_moep.py", ckpt_path)
                    except:
                        pass
                    
                    if self.api and self.repo_id:
                        try:
                            branch_name = f"checkpoint-{m_name}"
                            self.api.create_branch(repo_id=self.repo_id, branch=branch_name, exist_ok=True)
                            self.api.upload_folder(
                                repo_id=self.repo_id,
                                folder_path=ckpt_path,
                                path_in_repo=".",
                                repo_type="model",
                                revision=branch_name,
                                commit_message=f"MoEP checkpoint at {m_name} words"
                            )
                            print(f"Pushed to branch: {branch_name}")
                        except Exception as e:
                            print(f"Push failed: {e}")
                    
                    self.saved_milestones.add(m_name)
                    self.next_milestone_idx += 1
        return control

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss 

        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # Alignment check
        if outputs_student.logits.size() != avg_teacher_logits.size():
             min_vocab = min(outputs_student.logits.size(-1), avg_teacher_logits.size(-1))
             s_logits = outputs_student.logits[..., :min_vocab]
             t_logits = avg_teacher_logits[..., :min_vocab]
        else:
             s_logits = outputs_student.logits
             t_logits = avg_teacher_logits

        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(s_logits / self.args.temperature, dim=-1),
                F.softmax(t_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

# ============================================================
# Execution
# ============================================================

my_repo_id = "zhezhang-ovo/BabyLM-MoEP-Student"
api = HfApi()

training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=500,
    num_train_epochs=EPOCHS,        # [修改] 使用论文的 10 Epochs
    report_to=[],
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE, # [修改] 使用论文的 16
    save_total_limit=None,
    warmup_steps=WARMUP_STEPS,      # [修改] 使用论文的 800
    lr_scheduler_type="cosine",
    learning_rate=LR,               # [修改] 使用论文的 3e-4
    adam_beta1=ADAM_BETAS[0],       # [修改] Adam Beta1
    adam_beta2=ADAM_BETAS[1],       # [修改] Adam Beta2
    logging_steps=20,
    fp16=True, 
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
    save_safetensors=False, 
)

trainer = DistillationTrainer(
    student,
    training_args,
    teacher_models=teachers,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[WordMilestoneCB(
        tokenizer=tokenizer,
        api=api, 
        repo_id=my_repo_id, 
        seq_len=SEQ_LENGTH, 
        grad_acc=1, 
        bsz=BATCH_SIZE, 
        output_dir=MODEL_OUTPUT
    )]
)

trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
print(f"MoEP Training Finished! Saved to {MODEL_OUTPUT}")