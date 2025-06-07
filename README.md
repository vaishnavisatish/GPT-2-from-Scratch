# 🧠 GPT-2 from Scratch with Instruction Fine-Tuning

This project implements a GPT-2 language model entirely **from scratch** using PyTorch and pretrained weights. It includes the manual construction of all core components like multi-head attention, transformer blocks, GELU activation, and layer normalization. The model is fine-tuned on instruction-response pairs inspired by the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) and evaluated using [Ollama](https://ollama.com/), achieving an evaluation score of **80**.

---

## ✨ Features

- ✅ Manual implementation of GPT-2 architecture in PyTorch
- ✅ Custom tokenizer using OpenAI’s `tiktoken`
- ✅ Instruction-based fine-tuning using Alpaca-style prompts
- ✅ Configurable model sizes (small, medium, large, xl)
- ✅ Training, validation, and evaluation pipeline
- ✅ Generates coherent text using greedy decoding

---

## 🛠️ Model Architecture

The model was built using the following components:

- Multi-head Self-Attention
- Transformer Blocks with Residual Connections
- GELU Activation
- Layer Normalization
- Positional and Token Embeddings

### 🔧 Sample Configuration (GPT-2 Small)

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```
## Example Input Format 
```csharp
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Summarize the following paragraph.

### Input:
Artificial Intelligence is the field of developing machines that can perform tasks that typically require human intelligence...

### Response:
AI enables machines to perform human-like tasks such as learning, problem-solving, and decision-making.
```

### 📦 Installation
## 1. Clone the repository:
```bash
git clone https://github.com/vaishnavisatish/GPT-2-from-Scratch.git
cd GPT-2-from-Scratch
```
## 2. Install dependencies: 
```bash
pip install -r requirements.txt
```
## 3. (Optional) If using GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```



