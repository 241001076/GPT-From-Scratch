# 🧠 GPTMini – A Mini GPT Clone from Scratch

Welcome to **GPTMini**, a transformer-based language model built **entirely from scratch using PyTorch**!  
This project is a minimal, educational implementation of GPT-style models that helped me understand the internals of transformer architectures deeply.

---

## 📌 Features

- ✔️ Tokenizer integration
- ✔️ Positional Encoding
- ✔️ Input Embedding
- ✔️ Self-Attention mechanism
- ✔️ Residual connections + LayerNorm
- ✔️ FeedForward network
- ✔️ Stacked Transformer blocks
- ✔️ Full training pipeline from scratch
- ✔️ Text generation with sampling

---

## 🧱 Architecture Overview

Text → Tokenizer → Input Embedding →
→ [TransformerBlock × N] →
→ Linear layer (vocab projection) →
→ Generated Output

yaml
Copy
Edit

Each `TransformerBlock` includes:
- Self-Attention → Residual → LayerNorm  
- FeedForward → Residual → LayerNorm

---

## 📉 Training Log

Epoch 1/300, Loss: 134.3330
Epoch 51/300, Loss: 4.2212
Epoch 101/300, Loss: 1.1362
Epoch 151/300, Loss: 0.3862
Epoch 201/300, Loss: 0.3172
Epoch 251/300, Loss: 0.3834
Epoch 300/300, Loss: 0.2838

yaml
Copy
Edit

---

## 🔥 Sample Output

HHHHHa! and then the story continues with randomness based on data...

yaml
Copy
Edit

---

## 🛠️ How to Run

```bash
# Clone the repo
git clone https://github.com/241001076/GPT-From-Scratch.git
cd GPT-From-Scratch

# Make sure you have Python and PyTorch installed
pip install torch numpy

# Run the training
python new.py
🚀 What's Next?
✅ Add multi-head attention

✅ Add masked attention for autoregressive training

🚧 Improve tokenizer

🚧 Add text sampling UI

🤟 Credits
Made with 💻 + 🧠 + 🔥 by Harish Akshay H

🧠 Learnings
This project was built to:

Understand how GPT models are structured

Get hands-on with training loops, loss, optimizer

Explore transformers without libraries like HuggingFace

📄 License
MIT License — use it freely to learn and build more!

yaml
Copy
Edit

---

Machi, just save this as `README.md` in your repo folder, then run:

```bash
git add README.md
git commit -m "Add project README"
git push