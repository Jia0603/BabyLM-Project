# Weekly Report (Nov 11 – Nov 18)

## 1. Baseline Models: Architectures and Interaction Mechanisms

### 1.1 SimPO Baseline (BabyLM Interaction Track Official Baseline)

**Teacher Model:**  
- LLaMA-3 8B Instruct  
- Used as an **LLM-as-a-judge** scoring model  
- Provides scalar reward based on coherence, repetition, instruction-following, and style

**Student Model:**  
- GPT-2 Small (124M), decoder-only transformer  
- Policy updated via **SimPO**, a simplified PPO algorithm

**Interaction Pipeline:**  
1. Student generates text (story continuation)  
2. Teacher (LLaMA 3 8B) evaluates the output and gives a reward (0–10 or normalized)  
3. SimPO updates the Student’s policy toward higher-reward outputs  
4. Repeat over multiple rollouts

This baseline performs **RLHF-style optimization** but with a small student and a large instruction-following teacher.

---

### 1.2 "Once Upon a Time" Baseline (Story-Generation Example)

**Teacher Model:**  
- LLaMA-3 8B Instruct  
- Acts as a **story-quality scorer**, rating narrative coherence, creativity, grammar, and non-repetition

**Student Model:**  
- GPT-2 Small (124M)

**Interaction Pipeline:**  
1. Student produces a story continuation  
2. Teacher (LLaMA 8B) scores story quality  
3. Student updated using **PPO**, instead of SimPO  
4. Designed specifically for narrative story generation tasks

Difference vs SimPO:  
- Same teacher model  
- Different reward criteria  
- Different policy optimization (PPO vs SimPO)

---

### 1.3 BLM (Best Text-Only Baseline)

**Model:**  
- GPT-2 Medium/Small variant, improved tokenizer (SentencePiece Unigram), Adafactor optimizer  
- **Not** an interaction model; purely text-only LM  
- Serves as the strongest non-interaction reference

---

## 2. Our Data and Preprocessing Pipeline

### 2.1 Dataset
- BabyLM 10M training data  
- Train file: `corpus_split/train_babylm.txt`  
- Validation file: `corpus_split/val_babylm.txt`  

### 2.2 Tokenization
- We use the **original GPT-2 tokenizer (50257 vocab)**  
- Avoids breaking pretrained GPT-2 embeddings  
- teacher and student share the same tokenizer → essential for KL distillation

### 2.3 CustomDataset Processing
Steps:
1. Load full text into a single long string  
2. Replace newlines with spaces for smoother segments  
3. Slice into fixed-length chunks of **128 tokens**
4. For training:
   - `random_chunk=True` → random continuous segments improve data diversity  
5. For validation:
   - deterministic slicing (`random_chunk=False`)

### 2.4 Data Collator
- `DataCollatorForLanguageModeling(mlm=False)`  
- Pure causal LM: auto-shifts labels, auto-pads sequences

---

## 3. Fine-Tuning the Teacher Model (GPT-2 Large)

### 3.1 Teacher Architecture
- GPT-2 Large (774M parameters)  
- 36 layers, hidden size 1280, 20 attention heads  
- Loaded with full pretrained weights

### 3.2 Fine-Tuning Setup
- Learning rate: **2.5e-4**, cosine decay  
- Batch size: 128 (with gradient accumulation)  
- FP16 training  
- Warmup: 300 steps  
- Epochs: 3–4  
- Dataset: BabyLM 10M  
- Save the best model via validation loss

### 3.3 Teacher Performance
| Model | Eval Loss | Perplexity |
|-------|-----------|------------|
| Original GPT-2 Large | 3.1482 | 23.29 |
| Fine-tuned Teacher | **3.1096** | **22.41** |

This is a **small but expected** improvement:
- GPT-2 Large is already strong  
- BabyLM domain shift is small  
- Our goal is domain adaptation (not full pretraining)

The fine-tuned model now serves as a **BabyLM-specific expert** for distillation.

---

## 4. Student Distillation and CE Model Training

### 4.1 Student Model (GPT-2 Small)
- Random-initialized GPT-2 Small  
- 12 layers, 768 hidden size, 12 heads  
- Same tokenizer (50257)  

### 4.2 Teacher for Distillation
- The fine-tuned GPT-2 Large provides **logits (full probability distributions)**  
- Unlike SimPO or Once-Upon-a-Time, we do **no RL**  
- Distillation is purely supervised

---

## 4.3 Distillation Mechanism (Knowledge Distillation)

### Hard Targets (CE loss)
- Ground-truth next token  
- One-hot representation  
- Cross-entropy encourages correctness  
- Standard LM training

### Soft Targets (Teacher logits)
- Teacher produces probability over all tokens  
- Student tries to match teacher's distribution  
- Contains richer information:
  - synonyms  
  - semantic preferences  
  - uncertainty  
  - smoother gradients

### Temperature (T=2)
- Softens teacher probabilities:
$$
p_i^{(T)} = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

### KL Divergence Loss
$$
\mathcal{L}_{KD} = KL\left(p_T^{(T)} \,\Vert\, p_S^{(T)}\right)\cdot T^2
$$

### Final Combined Loss
$$
\mathcal{L} = \alpha \cdot L_{\text{CE}} + (1 - \alpha)\cdot L_{\text{KD}}
$$
- We use α = 0.5  
- Balance hard token correctness and soft-teacher knowledge

---

## 4.4 Pure CE-Only Model (Baseline)
- Student trained without teacher  
- Learning signal = pure cross-entropy  
- Used as baseline to evaluate KD effectiveness  

---

## 5. Summary of Progress This Week (11.11–11.18)

- Studied official Interaction baselines (SimPO & Once Upon a Time)  
- Clarified Teacher=LLAMA 3 8B Instruct for both baselines  
- Implemented full BabyLM data pipeline with GPT-2 tokenizer  
- Fine-tuned GPT-2 Large on BabyLM 10M  
- Evaluated finetuned teacher and compared with base GPT-2 Large  
- Implemented KD Trainer: CE + KL with temperature  
- Built two students: CE-only and Distilled  
- Tested evaluation pipeline & perplexity computation
