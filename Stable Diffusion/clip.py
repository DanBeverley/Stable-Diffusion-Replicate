# ---ROLE IN STABLE DIFFUSION---

# Conditioning Input : Encode texual descriptions into embeddings that can be used as conditioning input for cross attention

# Cross Attention Guidance : Text embeddings are used in Cross Attention layers to guide the generation of the image
# to make sure it aligns well with the texual description

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self , n_vocab:int , n_embd:int , n_token:int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab , n_embd)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((n_token , n_embd)))
    def forward(self ,token):
        # (Batch_size , Seq_len) -> (Batch_size , Seq_len , Dim)
        x = self.token_embedding(token)
        
        # (Batch_size , Seq_len) -> (Batch_size , Seq_len , Dim)
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self , n_head : int , n_embd : int):
        super().__init__()

        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head , n_embd)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward Layer
        self.linear_1 = nn.Linear(n_embd , 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd , n_embd)
    def forward(self , x):
        # (Batch_size , Seq_len , DIm)
        residue = x

        ### SELF ATTENTION ###

        # (Batch_size , Seq_len , DIm) -> (Batch_size , Seq_len , Dim)
        x = self.layernorm_1(x)
        # (Batch_size , Seq_len , Dim) -> (Batch_size , Seq_len , Dim)
        x = self.attention(x, casual_mask = True)

        # (Batch_size , Seq_len , Dim) + (Batch_size , Seq_len , Dim) -> (Batch_size , Seq_len , Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension

        residue = x
        # (Batch_size , Seq_len , Dim) -> (Batch_size , Seq_len , Dim)
        x = self.layernorm_2(x)

        # (Batch_size , Seq_len , Dim) -> (Batch_size , Seq_len , 4*Dim)
        x = self.linear_1(x)

        # (Batch_size , Seq_len , 4*Dim) -> (Batch_size , Seq_len , 4*Dim)
        x = x*torch.sigmoid(1.702*x)     # QuickGELU activation function

        # (Batch_size , Seq_len , 4*Dim) -> (Batch_size , Seq_len , Dim)
        x = self.linear_2(x)

        # (Batch_size , Seq_len , Dim) + (Batch_size , Seq_len , Dim) -> (Batch_size , Seq_len , Dim)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408 , 768 , 77)
        self.layers = nn.ModuleList([CLIPLayer(12,768) for i in range (12)])
        self.layernorm = nn.LayerNorm(768)
    def forward(self  , tokens:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_size , Seq_len) -> (Batch_size , Seq_len , Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to Transformer Encoder
        for layer in self.layers:
            # (Batch_size , Seq_len , Dim) -> (Batch_size , Seq_len , DIm)
            state = layer(state)
        # (Batch_size , Seq_len , Dim) -> (Batch_size , Seq_len , Dim)
        output = self.layernorm(state)
        return output