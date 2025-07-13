class Tokenizer():
    def __init__(self,text):
        self.x = sorted(list(set(text)))
        self.vocab = len(self.x)
        self.y = {ch: i for i, ch in enumerate(self.x)}
        self.z = {i: ch for ch, i in self.y.items()}

    def encode(self,a):
        return [self.y[c] for c in a]
    
    def decode(self,b):
        return ''.join([self.z[c] for c in b])


from Learning import Tokenizer
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

with open ("data.txt","r",encoding="utf-8") as f:
    text = f.read()

word = Tokenizer(text)
data = word.encode(text)
block_size = 8
X = []
Y = []
for i in range(len(data) - block_size):
    x = data[i:i+block_size]
    y = data[i+1:i+block_size+1]
    X.append(x)
    Y.append(y)

X = torch.tensor(X)
Y = torch.tensor(Y)

def positional_encoding(seq_len,d_model):
    PE = np.zeros((seq_len,d_model))
    for pos in range(seq_len):
        for i in range(0,d_model,2):
            angle = pos/(10000 ** ((2*i)/d_model))
            PE[pos,i] = np.sin(angle)
            if i+1 < d_model:
                PE[pos,i+1] = np.cos(angle)
    return torch.tensor(PE, dtype=torch.float32)

class Input_Embedding(nn.Module):
    def __init__(self,vocab,block_size,d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab,d_model)
        pe = positional_encoding(block_size,d_model).unsqueeze(0)
        self.register_buffer("positional_encoding",pe)

    def forward(self,x):
        token_embed = self.token_embedding(x)
        position_embed = self.positional_encoding[:, :x.size(1), :]
        return token_embed + position_embed

class self_attention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)

    def forward(self,x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2,-1)/ math.sqrt(self.d_model))
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,d_model,hidden_dim):
        super().__init__()
        self.attn = self_attention(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)
    def forward(self,x):
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.attn_norm(x)

        ff_out = self.ff(x)
        x = x + ff_out
        x = self.ff_norm(x)
        

class GPTMini(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, hidden_dim, n_layers):
        super().__init__()

        # 1. Input Embedding (token + position)
        self.embedding = Input_Embedding(vocab_size, block_size, d_model)

        # 2. Transformer Blocks (stacked)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, hidden_dim) for _ in range(n_layers)
        ])

        # 3. Final output layer → vocab size
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)     
        x = self.blocks(x)        
        logits = self.lm_head(x)  
        return logits