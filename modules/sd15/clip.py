import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention




class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab:int, n_embed: int, n_tokens: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))
        
    def forward(self, tokens):
        # B Seq
        
        x = self.token_embedding(tokens)
        x += self.position_embedding
        
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B Seq d_mod
        
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue
        
        # FF
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU 
        x = self.linear_2(x)
        x += residue

        return x
        
        


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)  # vocab_size d_model max_length
        
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)  # 12 слоев по 12 голов
        ])

        self.layernorm = nn.LayerNorm(768)


    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        state = self.embedding(tokens)  # B Seq 768
        
        for layer in self.layers:
            state = layer(state)
            
        output = self.layernorm(state)
        
        return output  # B Seq 768
    
    
    





