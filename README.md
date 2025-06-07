# GPT-2-from-Scratch-with-Instruction-Tuning
Built a GPT-2 language model entirely from scratch using PyTorch and pretrained weights. All core components including multi-head self-attention, transformer blocks, and layer normalization were implemented manually. The model was fine-tuned using instruction-based data (Alpaca dataset) and evaluated using Ollama with a score of 80.
## 🔧 Key Features
Manual implementation of:
Multi-head Attention
Transformer Blocks
GELU Activation and LayerNorm
Positional and Token Embeddings
Uses OpenAI's tiktoken for GPT-2 tokenization
Instruction fine-tuning using Alpaca-style prompts
Full training and evaluation pipeline
Achieved a score of 80 on Ollama
