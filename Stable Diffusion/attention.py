# ---ROLE IN STABLE DIFFUSION---
# Self Attention and Cross Attention guide the reverse process (denoising) to reconstruct image in a manner
# consistent with the conditioning input

# Self Attention maintain a hierachical understanding of image structure 
# Cross Attention ensures the generated image aligns with external conditions

import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module): # Q,K,V projects Image features
    def __init__ (self , n_heads , d_embed , in_proj_bias = True , out_proj_bias = True):
        super().__init__()
    # Combines Wq , Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed , 3*d_embed , bias = in_proj_bias)
    # Represents the Wo matrix
        self.out_proj = nn.Linear(d_embed , d_embed , bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self , x , casual_mask = False):
        # x : # (Batch_size , Seq_Len , Dim)

        # (Batch_Size , Seq_Len , Dim)
        input_shape = x.shape

        # (Batch_Size , Seq_Len , Dim)
        batch_size , sequence_length , d_embed = input_shape

        # (Batch_size , Seq_len , H , Dim/H)
        interim_shape = (batch_size , sequence_length , self.n_heads , self.d_head)

        # (Batch_size , Seq_len , DIm) -> (Batch_Size , Seq_Len , DIm*3) -> 3 tensor of shape (Batch_size , Seq_len , Dim)
        q,k,v = self.in_proj(x).chunk(3 , dim =-1)
        
        # (Batch_size , Seq_len , DIm) -> (Batch_size , Seq_len , H , Dim/H) -> (Batch_size , H , Seq_len , DIm/H)
        q = q.view(interim_shape).tranpose(1,2)
        k = k.view(interim_shape).traspose(1,2)
        v = v.view(interim_shape).tranpose(1,2)

        # (Batch_size , H , Seq_len , Dim/H) @ (Batch_size , H , Dim/H , Seq_len) -> (Batch_size , H , Seq_len , Seq_len)
        weight = q @ k .tranpose(-1,-2)

        if casual_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight , dtype = torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask , -torch.inf)
        
        # Divide  by d_k (Dim /H)
        # (Batch_size , H , Seq_len , Seq_len) -> (Batch_size , H , Seq_len , Seq_len)
        weight /= math.sqrt(self.d_head)

        # (Batch_size , H , Seq_len , Seq_len) -> (Batch_size , H , Seq_len , Seq_len)
        weight = F.softmax(weight , dim =-1)

        # (Batch_size , H , Seq_len , Seq_len) @ (Batch_size , H , Seq_len , Dim/H) -> (Batch_size , H , Seq_len , Dim/H)
        output = weight @ v

        # (Batch_size , H , Seq_len , Dim/H) -> (Batch_size , Seq_len , H , DIm/H)
        output = output.transpose(1,2)
        
        # (Batch_size , Seq_len , H , Dim/H) -> (Batch_Size , Seq_len , Dim)
        output = output.reshape(input_shape)

        # (Batch_size , Seq_len , DIm) -> (Batch_size , Seq_len , Dim)
        output = self.out_proj(output)

        # (Batch_size , Seq_len , Dim)
        return output

class CrossAttention(nn.Module): # Q: Projects image features . K , V : Projects Conditional Information (Text)
    def __init__(self , n_heads , d_embed , d_cross , in_proj_bias = True , out_proj_bias = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed , d_embed , bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross , d_embed , bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross , d_embed , bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed , d_embed , bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads
    def forward(self , x , y):
        # x (latent) : # (Batch_size , Seq_len_Q , Dim_Q)
        # y (context) : # (Batch_Size , Seq_Len_KV , Dim_KV) = (Batch_size , 77 , 768)
        input_shape = x.shape
        batch_size , sequence_length , d_embed = input_shape

        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = DIm_Q
        interim_shape = (batch_size , -1 , self.n_heads , self.d_head)

        # (Batch_size , Seq_len_Q , Dim_Q) -> (Batch_size , Seq_len_Q , Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size , Seq_Len_KV , Dim_kv) -> (Batch_size , Seq_len_kv , DIm_Q)
        k = self.k_proj(y)
        # (Batch_Size , Seq_len_kv , Dim_kv) -> (Batch_size , Seq_len_kv , Dim_Q)
        v = self.v_proj(y)

        # (Batch_size , Seq_len_Q , Dim_Q) -> (Batch_size , Seq_len_Q , H , Dim_Q / H) -> (Batch_size , H , Seq_len_Q , Dim_Q / H)
        q = q.view(interim_shape).tranpose(1,2)
        # (Batch_size , Seq_len_KV , DIm_Q) -> (Batch_size , Seq_len_KV , H , Dim_Q / H) -> (Batch_size , H , Seq_len_KV , Dim_Q/H)
        k = k.view(interim_shape).tranpose(1,2)
        # (Batch_size , Seq_len_KV , Dim_Q) -> (Batch_size , Seq_len_KV , H , Dim_Q/H) -> (Batch_size , H , Seq_len_KV , DIm_Q/H)
        v = v.view(interim_shape).tranpose(1,2)

        # (Batch_size , H , Seq_len_Q , Dim_Q / H) @  (Batch_size , H , Dim_Q / H, Seq_len_KV) -> (Batch_Size , H , Seq_len_Q , Seq_len_KV)
        weight = q @ k.tranpose(-1,-2)

        # (Batch_size , H , Seq_len_Q , Seq_len_KV)
        weight /= math.sqrt(self.d_head)

        # (Batch_size , H , Seq_len_Q , Seq_len_KV)
        weight = F.softmax(weight , dim = -1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # (Batch_size , H , Seq_len_Q , Dim_Q /H) -> (Batch_size , Seq_len_Q , H , Dim_Q/H)
        output = output.traspose(1,2).contiguous()

        # (Batch_size , Seq_len_Q , H , Dim_Q/H) -> (Batch_size , Seq_len_Q , Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_size , Seq_len_Q , Dim_Q) -> (Batch_size , Seq_len_Q , Dim_Q)
        output = self.out_proj(output)

        # (Batch_size , Seq_len_Q , Dim_Q)
        return output