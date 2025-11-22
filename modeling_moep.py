import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union 

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

@dataclass
class BaseModelOutputWithAux(BaseModelOutput):
    aux_loss: Optional[torch.Tensor] = None

def create_causal_mask(seq_len: int, device=None, dtype=torch.float32):
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask

def create_sw_mask(seq_len: int, window_size: Optional[int], device=None, dtype=torch.float32):
    if not window_size:
        return None
    i = torch.arange(seq_len, device=device)
    j = torch.arange(seq_len, device=device)
    allowed = (j[None, :] <= i[:, None]) & (j[None, :] >= i[:, None] - (window_size - 1))
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask

class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, b: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=b)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

class Layer_norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, b: bool = True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=b)

    def forward(self, x: torch.Tensor):
        return self.layer_norm(x)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dim_out: Optional[int] = None, b: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=b)
        out_dim = in_dim if dim_out is None else dim_out
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=b)

    @staticmethod
    def gelu_new(x: torch.Tensor):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.gelu_new(x)
        x = self.linear2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.Q = nn.Linear(dim, dim, bias=True)
        self.K = nn.Linear(dim, dim, bias=True)
        self.V = nn.Linear(dim, dim, bias=True)
        self.O = nn.Linear(dim, dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, *, causal_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        B, S, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        q = self.Q(x).view(B, S, H, Dh).transpose(1, 2)
        k = self.K(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.V(x).view(B, S, H, Dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        neg_inf = torch.finfo(attn.dtype).min

        if causal_mask is not None:
            attn = attn + causal_mask[None, None, :, :]

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], neg_inf)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)

        if key_padding_mask is not None:
            query_mask = (~key_padding_mask).to(out.dtype).unsqueeze(-1)
            out = out * query_mask

        return self.O(out)

class Layer_Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_multp: int, dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.norm_1 = Layer_norm(dim)
        self.norm_2 = Layer_norm(dim)
        self.attention = SelfAttention(dim, n_heads, attn_dropout)
        self.feedforward = MLP(dim, dim * ffn_multp)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, causal_mask=None, key_padding_mask=None):
        residual = x
        x = self.norm_1(x)
        x = self.attention(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = residual + x
        return x

class Top_k_Router(nn.Module):
    def __init__(self, dim: int, n_experts: int, k: int):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.router = Linear(dim, n_experts)

    def forward(self, x: torch.Tensor):
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(probs, k=self.k, dim=-1)
        topk_prob = topk_prob / (topk_prob.sum(dim=-1, keepdim=True) + 1e-9)

        p_mean = probs.mean(dim=0)
        T, E = probs.size()
        one_hot = F.one_hot(topk_idx, num_classes=E).sum(dim=1)
        f_mean = one_hot.float().mean(dim=0) / float(self.k)
        aux_loss = (E * (p_mean * f_mean).sum())

        return topk_idx, topk_prob, aux_loss

class MoE_Block(nn.Module):
    def __init__(self, dim_in: int, n_experts: int, k: int, dim_hidden: int, dim_out: Optional[int] = None):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.dim_out = dim_in if dim_out is None else dim_out
        self.router = Top_k_Router(dim_in, n_experts, k)
        self.experts = nn.ModuleList([nn.Linear(dim_in, self.dim_out, bias=True) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        T = B * S
        x_flat = x.view(T, D)

        topk_idx, topk_prob, aux_loss = self.router(x_flat)
        y_flat = x_flat.new_zeros((T, self.dim_out))

        for slot in range(self.k):
            e_ids = topk_idx[:, slot]
            e_prob = topk_prob[:, slot].unsqueeze(-1)
            for e in range(self.n_experts):
                idx = (e_ids == e).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                y_e = self.experts[e](x_flat.index_select(0, idx))
                y_flat.index_add_(0, idx, y_e * e_prob.index_select(0, idx))

        return y_flat.view(B, S, self.dim_out), aux_loss

class ParallelLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_blocks: int, k: int, ffn_multp: int, dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([Layer_Block(dim, n_heads, ffn_multp, dropout, attn_dropout) for _ in range(n_blocks)])
        self.router = Top_k_Router(dim, n_blocks, k)

    def forward(self, x: torch.Tensor, *, causal_mask=None, key_padding_mask=None):
        B, S, D = x.shape
        T = B * S
        outs = [blk(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask) for blk in self.blocks]
        
        Y = torch.stack([y.view(T, D) for y in outs], dim=1)

        x_flat = x.view(T, D)
        topk_idx, topk_prob, aux_loss = self.router(x_flat)

        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        Y_topk = torch.gather(Y, dim=1, index=gather_idx)
        y_flat = (Y_topk * topk_prob.unsqueeze(-1)).sum(dim=1)

        return y_flat.view(B, S, D), aux_loss

class MoEPLMConfig(PretrainedConfig):
    model_type = "moep_lm"

    def __init__(
        self,
        vocab_size: int = 16384,
        bos_token_id = 1,
        eos_token_id = 2,
        pad_token_id = 3,
        mask_token_id = 4,
        max_position_embeddings: int = 1024,
        n_layers: int = 12,
        ffn_multp: int = 4,
        dim: int = 384, # Adjusted from 348 (likely typo) to 384 for standard head size
        n_heads: int = 6,
        n_parallel_blocks: int = 4,
        parallel_dim: int = 192,
        parallel_n_heads: int = 3,
        parallel_k: int = 2,
        sliding_window: Optional[int] = None,
        n_experts: int = 4,
        moe_k: int = 2,
        attn_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layers = n_layers
        self.ffn_multp = ffn_multp
        self.dim = dim
        self.n_heads = n_heads
        self.n_parallel_blocks = n_parallel_blocks
        self.parallel_dim = parallel_dim
        self.parallel_n_heads = parallel_n_heads
        self.parallel_k = parallel_k
        self.sliding_window = sliding_window
        self.n_experts = n_experts
        self.moe_k = moe_k
        self.attn_pdrop = attn_pdrop
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_word_embeddings = tie_word_embeddings

class MoEPModel(PreTrainedModel):
    config_class = MoEPLMConfig
    base_model_prefix = "model"

    def __init__(self, config: MoEPLMConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.dim)
        self.embd_dropout = nn.Dropout(config.embd_pdrop)

        self.layers = nn.ModuleList()
        # Large Layer
        self.layers.append(Layer_Block(config.dim, config.n_heads, config.ffn_multp, dropout=config.resid_pdrop, attn_dropout=config.attn_pdrop))
        # MoE Linear Shrink
        self.layers.append(MoE_Block(config.dim, config.n_experts, config.moe_k, dim_hidden=config.dim * 2, dim_out=config.parallel_dim))
        # Parallel Layers
        for _ in range(config.n_layers - 2):
            self.layers.append(ParallelLayer(config.parallel_dim, config.parallel_n_heads, config.n_parallel_blocks, config.parallel_k, config.ffn_multp, dropout=config.resid_pdrop, attn_dropout=config.attn_pdrop))
        # MoE Linear Grow
        self.layers.append(MoE_Block(config.parallel_dim, config.n_experts, config.moe_k, dim_hidden=config.parallel_dim * config.ffn_multp, dim_out=config.dim))
        # Large Layer
        self.layers.append(Layer_Block(config.dim, config.n_heads, config.ffn_multp, dropout=config.resid_pdrop, attn_dropout=config.attn_pdrop))

        self.norm = Layer_norm(config.dim)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, return_dict: bool = True, return_aux: Optional[bool] = False, return_aux_dict: Optional[bool] = False, **unused) -> Union[BaseModelOutputWithAux, BaseModelOutput, Tuple]:
        device = input_ids.device
        B, S = input_ids.shape
        key_padding = input_ids.eq(self.config.pad_token_id) if attention_mask is None else (attention_mask == 0)

        x = self.embed_tokens(input_ids)
        mask_dtype = x.dtype
        causal = create_causal_mask(S, device=device, dtype=mask_dtype)
        sw = create_sw_mask(S, self.config.sliding_window, device=device, dtype=mask_dtype)
        sw_mask = causal if sw is None else torch.minimum(causal, sw)
        
        pos = torch.arange(S, device=device)    
        pos = pos.unsqueeze(0).expand(B, S)
        x = x + self.embed_positions(pos)
        x = self.embd_dropout(x)

        aux_total = None

        # First large
        x = self.layers[0](x, causal_mask=causal, key_padding_mask=key_padding)
        # MoE shrink
        x, aux = self.layers[1](x)
        if aux is not None: aux_total = aux

        idx = 2
        for i in range(self.config.n_layers - 2):
            use_sw = (self.config.sliding_window is not None) and (i % 2 == 1)
            mask = sw_mask if use_sw else causal
            x, aux = self.layers[idx](x, causal_mask=mask, key_padding_mask=key_padding)
            if aux is not None: aux_total = aux_total + aux if aux_total is not None else aux
            idx += 1

        # MoE grow
        x, aux = self.layers[idx](x)
        if aux is not None: aux_total = aux_total + aux if aux_total is not None else aux
        idx += 1

        # Second large
        x = self.layers[idx](x, causal_mask=causal, key_padding_mask=key_padding)
        last_hidden = self.norm(x)

        if not return_dict:
            return (last_hidden, aux_total) if return_aux else (last_hidden,)

        if return_aux_dict:
            return BaseModelOutputWithAux(last_hidden_state=last_hidden, hidden_states=None, attentions=None, aux_loss=aux_total)
        else:
            return BaseModelOutput(last_hidden_state=last_hidden, hidden_states=None, attentions=None)

class MoEPForCausalLM(PreTrainedModel):
    config_class = MoEPLMConfig
    base_model_prefix = "model"

    def __init__(self, config: MoEPLMConfig):
        super().__init__(config)
        self.config = config
        self.model = MoEPModel(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
            
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.LongTensor] = None, return_dict: bool = True, return_aux: Optional[bool] = False, aux_loss_weight: Optional[float] = 0.2, **unused) -> Union[CausalLMOutput, Tuple]:
        if return_aux:
            base_out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_aux_dict=True, return_aux=True)
        else: 
            base_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
    
        logits = self.lm_head(base_out.last_hidden_state).float()
        loss = None
        
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            pad_id = self.config.pad_token_id
            if pad_id is not None:
                shift_labels = shift_labels.masked_fill(shift_labels == pad_id, -100)
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            
            # Auto add aux loss for router load balancing
            aux = getattr(base_out, "aux_loss", None)
            if aux is not None:
                loss = loss + aux_loss_weight * aux

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutput(loss=loss, logits=logits, hidden_states=base_out.hidden_states, attentions=base_out.attentions)

# Register the config and model
AutoConfig.register("moep_lm", MoEPLMConfig)
AutoModelForCausalLM.register(MoEPLMConfig, MoEPForCausalLM)
AutoModel.register(MoEPLMConfig, MoEPModel)