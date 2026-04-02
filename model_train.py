from collections import Counter
import re
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import multiprocessing as mp
import torch.utils.checkpoint as cp
from collections import Counter
from torch.amp import autocast, GradScaler
import os
import requests
import sentencepiece as spm
import cudf
from torch.utils.checkpoint import checkpoint
from concurrent.futures import ThreadPoolExecutor
import sentencepiece as spm
import torch
from time import perf_counter
class SPTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        
        self.vocab_size_actual = self.sp.get_piece_size()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0

    
    def encode(self, text):
        
        return self.sp.encode(text, out_type=int)

    
    def encode_batch(self, texts):
        
        return self.sp.encode(texts, out_type=int)

    
    def decode(self, ids):
        return self.sp.decode(ids)

    def decode_batch(self, batch_ids):
        return self.sp.decode(batch_ids)


def apply_rope(x, seq_dim=1):
    B, T, H, D = x.shape
    half = D // 2

    freq = torch.arange(half, device=x.device, dtype=x.dtype)
    freq = 1.0 / (10000 ** (freq / half))

    pos = torch.arange(T, device=x.device, dtype=x.dtype)
    angles = torch.einsum("t,d->td", pos, freq)

    sin = angles.sin()[None, :, None, :]
    cos = angles.cos()[None, :, None, :]

    x1 = x[..., :half]
    x2 = x[..., half:]

    x_rot = torch.cat([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return x_rot

def build_online_samples(sentence, tok, window=16):
    pad = tok.pad_id
    ids = tok.encode(sentence)
    X, Y = [], []
    for i in range(1, len(ids)):
        ctx = ids[max(0, i - window):i]
        ctx = [pad] * (window - len(ctx)) + ctx
        nxt = ids[i]
        X.append(ctx)
        Y.append(nxt)
    if not X:
        return torch.empty(0, window, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def grow_embedding(old_emb, new_vocab_size):
    old_weight = old_emb.weight.data
    old_vocab, dim = old_weight.shape

    if new_vocab_size <= old_vocab:
        return old_emb

    new_emb = nn.Embedding(new_vocab_size, dim)
    new_emb.weight.data[:old_vocab] = old_weight.clone()
    nn.init.normal_(new_emb.weight.data[old_vocab:], mean=0.0, std=0.02)
    return new_emb


def compute_surprise(logits, target_id):
    probs = torch.softmax(logits, dim=-1)
    p = float(probs[0, target_id].item())
    confidence = p
    surprise = 1.0 - confidence
    return confidence, surprise
class RelationalWorldModel(nn.Module):
    def __init__(self, dim=256, max_nodes=512, ttl=64):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.default_ttl = ttl
        self.register_buffer("nodes", torch.zeros(0, dim))
        self.node_ttl = []
        self.W_msg = nn.Linear(dim * 2, dim)
        self.W_self = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def query(self, vec, k=4):
        if self.nodes.size(0) == 0:
            return torch.zeros(self.dim, device=vec.device)
        if vec.dim() == 3:
            vec = vec.mean(dim=1)
        if vec.dim() == 2:
            vec = vec.mean(dim=0)
        if vec.dim() == 1 and vec.shape[0] != self.dim:
            return torch.zeros(self.dim, device=vec.device)
        scores = torch.matmul(self.nodes, vec.to(self.nodes.device))
        topk = torch.topk(scores, min(k, scores.size(0)), dim=0).indices
        return self.nodes[topk].mean(dim=0)
    def _add_node(self, vec):
        with torch.no_grad():
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)
            elif vec.dim() == 3:
                vec = vec.mean(dim=0)

            vec = vec.detach()

            if self.nodes.size(0) >= self.max_nodes:
                self.nodes = self.nodes[1:]
                self.node_ttl = self.node_ttl[1:]

            self.nodes = torch.cat([self.nodes, vec.to(self.nodes.device)], dim=0)
            self.node_ttl.append(self.default_ttl)

    def store(self, subj, rel, obj):
        with torch.no_grad():
            vec = subj + rel + obj
            vec = vec.mean(dim=0).detach()
            self._add_node(vec)

    def decay(self):
        if len(self.node_ttl) == 0:
            return
        self.node_ttl = [t - 1 for t in self.node_ttl]
        mask = torch.tensor([t > 0 for t in self.node_ttl], device=self.nodes.device)
        self.nodes = self.nodes[mask]
        self.node_ttl = [t for t in self.node_ttl if t > 0]

class RelationalGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, concept_mix):
        return torch.sigmoid(self.gate(concept_mix))

class ConceptExtractor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.subj = nn.Linear(dim, dim)
        self.act  = nn.Linear(dim, dim)
        self.obj  = nn.Linear(dim, dim)

        self.ws = nn.Linear(dim, 1)
        self.wa = nn.Linear(dim, 1)
        self.wo = nn.Linear(dim, 1)

    def forward(self, h):
        w_s = torch.softmax(self.ws(h).squeeze(-1), dim=-1)
        w_a = torch.softmax(self.wa(h).squeeze(-1), dim=-1)
        w_o = torch.softmax(self.wo(h).squeeze(-1), dim=-1)

        subj = torch.bmm(w_s.unsqueeze(1), self.subj(h)).squeeze(1)
        act  = torch.bmm(w_a.unsqueeze(1), self.act(h)).squeeze(1)
        obj  = torch.bmm(w_o.unsqueeze(1), self.obj(h)).squeeze(1)

        return subj, act, obj

class ConfidenceNet(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size + vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act = nn.ReLU()

    def forward(self, h, logits):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = logits.clamp(-20.0, 20.0)
        probs = torch.softmax(logits.float(), dim=-1)
        x = torch.cat([h, probs], dim=-1)
        x = self.act(self.fc1(x))
        conf = torch.sigmoid(self.fc2(x))
        return conf

class WorkingMemory(nn.Module):
    def __init__(self, num_slots=12, slot_dim=256):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.init_content = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.01)
        self.addr_proj = nn.Linear(slot_dim, num_slots)
        self.write_proj = nn.Linear(slot_dim, slot_dim)
        self.write_gate = nn.Linear(slot_dim, 1)
        self.norm = RMSNorm(slot_dim)

    def init_state(self, batch_size, device):
        return self.init_content.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def forward(self, query, wm_state):
        B, D = query.shape
        if wm_state is None:
            wm_state = self.init_state(B, query.device)
        attn = torch.softmax(torch.bmm(wm_state, query.unsqueeze(-1)).squeeze(-1), dim=-1)
        read = torch.bmm(attn.unsqueeze(1), wm_state).squeeze(1)
        addr = torch.softmax(self.addr_proj(query), dim=-1)
        gate = torch.sigmoid(self.write_gate(query))
        write = torch.tanh(self.write_proj(query))
        delta = addr.unsqueeze(-1) * write.unsqueeze(1) * gate.unsqueeze(-1)
        new_state = self.norm(wm_state + delta)
        return read, new_state

class LiquidSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=2, eps=1e-6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model)

    def phi(self, x):
        return F.elu(x) + 1.0
    def forward(self, x):
        B, T, D = x.shape
        q = self.phi(self.q_proj(x).view(B, T, self.num_heads, self.head_dim))
        k = self.phi(self.k_proj(x).view(B, T, self.num_heads, self.head_dim))
        v =          self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        q = apply_rope(q)
        k = apply_rope(k)

        
        kv  = torch.einsum("bthd,bthe->bhde", k, v)   
        out = torch.einsum("bthd,bhde->bthe", q, kv)   
        z   = torch.einsum("bthd,bhd->bth",  q, k.sum(dim=1)).clamp(min=self.eps)

        out = (out / z.unsqueeze(-1)).reshape(B, T, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return gate * out + (1.0 - gate) * x

class LiquidGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.log_dt = nn.Parameter(torch.zeros(hidden_size))
        self.nonlinearity = nn.Tanh()
        self.W_z_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_z_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, h_t):
        tau = torch.exp(self.log_tau).unsqueeze(0)
        dt  = torch.exp(self.log_dt).unsqueeze(0)
        u = self.W_in(x_t) + self.W_h(h_t)
        f = self.nonlinearity(u)
        dh = -tau * h_t + f
        h_liquid = h_t + dt * dh
        z_x = self.W_z_x(x_t)
        z_h = self.W_z_h(h_t)
        z = self.sigmoid(z_x + z_h)
        h_next = (1.0 - z) * h_t + z * h_liquid
        return h_next



class S4DSSM(nn.Module):
    def __init__(self, d_state=64, d_model=64):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.log_lambda = nn.Parameter(-0.5 * torch.ones(d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.05)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.05)
        self.log_dt = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, x):
        B, T, D = x.shape
        
        dt      = torch.exp(self.log_dt)
        lambda_ = -torch.exp(self.log_lambda)          
        log_decay = lambda_ * dt                        

        t_idx = torch.arange(T, device=x.device, dtype=x.dtype)  

        
        log_decay_t = t_idx.unsqueeze(1) * log_decay.unsqueeze(0)  
        log_decay_t = log_decay_t.clamp(min=-30.0, max=0.0)        

        decay_t     = log_decay_t.exp()                  
        inv_decay_t = (-log_decay_t).exp().clamp(max=1e6) 

        Bu = x @ self.B.T                                

        Bu_scaled = Bu * inv_decay_t.unsqueeze(0)        
        h_scaled  = Bu_scaled.cumsum(dim=1)
        h_out     = h_scaled * decay_t.unsqueeze(0)

        return h_out @ self.C.T                          

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=128):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.ssm = S4DSSM(d_state=d_state, d_model=d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x_in = self.in_proj(x)
        x_ssm = self.ssm(x_in)
        x_out = x_in + x_ssm
        x_out = self.out_proj(x_out)
        x_out = self.norm(x_out)
        return x_out
def build_sequences_sp(corpus, tok, seq_len=64, max_sequences=2000000):
    X = []
    pad = tok.pad_id
    for line in corpus:
        ids = tok.encode(line)
        if len(ids) < 2:
            continue
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [pad] * (seq_len - len(ids))
        X.append(ids)
        if len(X) >= max_sequences:
            break
    if not X:
        return torch.empty(0, seq_len - 1, dtype=torch.long), torch.empty(0, seq_len - 1, dtype=torch.long)
    X = torch.tensor(X, dtype=torch.long)
    inputs = X[:, :-1]
    targets = X[:, 1:]
    return inputs, targets

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        a = self.w1(x)
        b = self.w2(x)
        x = F.silu(a) * b
        return self.w3(x)

class WorkingMemory(nn.Module):
    def __init__(self, num_slots=12, slot_dim=256):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.init_content = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.01)
        self.addr_proj = nn.Linear(slot_dim, num_slots)
        self.write_proj = nn.Linear(slot_dim, slot_dim)
        self.write_gate = nn.Linear(slot_dim, 1)
        self.norm = RMSNorm(slot_dim)

    def init_state(self, batch_size, device):
        return self.init_content.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def forward(self, query, wm_state):
        B, D = query.shape
        if wm_state is None:
            wm_state = self.init_state(B, query.device)
        attn = torch.softmax(torch.bmm(wm_state, query.unsqueeze(-1)).squeeze(-1), dim=-1)
        read = torch.bmm(attn.unsqueeze(1), wm_state).squeeze(1)
        addr = torch.softmax(self.addr_proj(query), dim=-1)
        gate = torch.sigmoid(self.write_gate(query))
        write = torch.tanh(self.write_proj(query))
        delta = addr.unsqueeze(-1) * write.unsqueeze(1) * gate.unsqueeze(-1)
        new_state = self.norm(wm_state + delta)
        return read, new_state
class LiquidLM(nn.Module):
    def __init__(self, vocab_size, dim=768, layers=12, rwm_dim=128):
        super().__init__()
        self.dim = dim
        self.rwm_dim = rwm_dim

        self.embed = nn.Embedding(vocab_size, dim)
        self.attn = LiquidSelfAttention(dim, num_heads=8)
        self.mamba = nn.ModuleList([MambaBlock(dim, d_state=64) for _ in range(layers)])

        self.norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, 4 * dim)

        self.concepts = ConceptExtractor(dim)
        self.subj_proj = nn.Linear(dim, rwm_dim)
        self.obj_proj = nn.Linear(dim, rwm_dim)
        self.rwm = RelationalWorldModel(dim=rwm_dim)
        self.rwm_proj = nn.Linear(dim, rwm_dim)

        self.wm = WorkingMemory(num_slots=8, slot_dim=rwm_dim)
        self.wm_back = nn.Linear(rwm_dim, dim)
        self.final_norm = RMSNorm(dim)

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)


    def forward(self, ids, wm_state=None):
        x = self.embed(ids)          
        x = checkpoint(self.attn, x)
        for layer in self.mamba:
            x = checkpoint(layer, x)

        x = self.norm(x)
        x = self.ff(x)

        subj, act, obj = self.concepts(x)
        subj_r = self.subj_proj(subj)    
        obj_r  = self.obj_proj(obj)      
        rel    = self.rwm_proj(act)      

        self.rwm.store(subj_r, rel, obj_r)
        self.rwm.decay()

        query_vec = subj_r
        rctx = self.rwm.query(query_vec)     
        
        if rctx.dim() == 1:
            rctx = rctx.unsqueeze(0).expand_as(subj_r)
        else:
            rctx = rctx.expand_as(subj_r)

        mix_r = subj_r + rctx               

        wm_read, wm_state = self.wm(mix_r, wm_state)  
        
        mix = x + self.wm_back(wm_read).unsqueeze(1)  
        mix = self.final_norm(mix)
        logits = self.lm_head(mix)
            
        return logits

class MirrorLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_size=64, window=16):
        super().__init__()
        self.window = window
        self.hidden_size = hidden_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = LiquidSelfAttention(d_model, num_heads=2)
        self.gru = nn.GRU(d_model, hidden_size, batch_first=True)  
        self.ln = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.attn(x)
        out, h_n = self.gru(x)               
        h = h_n.squeeze(0)
        pred_error = out[:, -1, :] - out[:, -2, :] if out.size(1) > 1 else torch.zeros_like(h)
        h = self.ln(h)
        h = self.dropout(h)
        logits = self.lm_head(h)
        pred_error_norm = pred_error.pow(2).mean().sqrt().item()
        return logits, h, pred_error_norm
class Hippocampus(nn.Module):
    def __init__(self, max_episodes=512):
        super().__init__()
        self.max_episodes = max_episodes
        self.episodes = [] 

    def store(self, sentence, priority):
        if len(self.episodes) > 0 and self.episodes[-1][0] == sentence:
            return

        self.episodes.append((sentence, float(priority)))

        self.episodes = sorted(self.episodes, key=lambda x: x[1], reverse=True)

        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[:self.max_episodes]




    def sample_batch(self, batch_size=8):
        if not self.episodes:
            return []
        weights = [p for (_, p) in self.episodes]
        total = sum(weights)
        if total <= 0:
            weights = [1.0 for _ in weights]
            total = sum(weights)
        probs = [w / total for w in weights]
        idxs = random.choices(range(len(self.episodes)), weights=probs, k=min(batch_size, len(self.episodes)))
        return [self.episodes[i][0] for i in idxs]
def consolidate_memory(real_model, mirror_model, hippocampus, tok,
                       window=16, device_real="cpu", rounds=50,
                       base_corpus=None):

    real_model.train()
    opt = torch.optim.AdamW(real_model.parameters(), lr=1e-5)

    for _ in range(rounds):
        batch = hippocampus.sample_batch(batch_size=4)
        if not batch:
            break

        for sentence in batch:
            Xb, Yb = build_sequences_sp(base_corpus, tok)
            before_loss = F.cross_entropy(
                real_model(Xb.to(device_real))[0],
                Yb.to(device_real)
            ).item()

            X, Y = build_online_samples(sentence, tok, window)
            if X.numel() == 0:
                continue

            X = X.to(device_real)
            Y = Y.to(device_real)

            opt.zero_grad()
            logits, _, _ = real_model(X)
            loss = F.cross_entropy(logits, Y)
            loss.backward()

            old_params = [p.clone() for p in real_model.parameters()]

            opt.step()

            after_loss = F.cross_entropy(
                real_model(Xb.to(device_real))[0],
                Yb.to(device_real)
            ).item()

            if after_loss > before_loss:
                with torch.no_grad():
                    for p, old in zip(real_model.parameters(), old_params):
                        p.copy_(old)

def learn_new_sentence(real_model, mirror_model, hippocampus,
                       sentence, tok, window=16, device="cpu",
                       adapter_steps=50, alpha=1.0):

    new_tokens = tok.observe(sentence)
    if new_tokens:
        new_vocab = tok.vocab_size_actual
        real_model.grow_vocab(new_vocab, tok)
        mirror_model.grow_vocab(new_vocab)

    X_new, Y_new = build_online_samples(sentence, tok, window)
    if X_new.numel() == 0:
        return

    X_new = X_new.to(device)
    Y_new = Y_new.to(device)

    base_sentences = [ep[0] for ep in hippocampus.episodes]


    if len(base_sentences) > 0:
        Xb, Yb = build_sequences_sp(base_sentences, tok)
        if Xb.numel() > 0:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
        else:
            Xb = Yb = None
    else:
        Xb = Yb = None

    real_model.train()
    adapter_params = [p for n, p in real_model.named_parameters() if "adapter" in n]
    opt = torch.optim.AdamW(adapter_params, lr=1e-3)

    for _ in range(adapter_steps):
        opt.zero_grad()

        logits_new, _ = real_model(X_new)
        loss_new = F.cross_entropy(logits_new, Y_new)
        if Xb is not None:
            logits_b, _ = real_model(Xb)
            loss_b = F.cross_entropy(logits_b, Yb)
            loss = loss_new + alpha * loss_b
        else:
            loss = loss_new

        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.softmax(logits_new, dim=-1)
        surpr = 1.0 - probs.gather(1, Y_new.unsqueeze(1)).mean().item()
        hippocampus.store(sentence, float(surpr ** 2))
def compute_accuracy(model, X, Y, device):
    model.eval()

    if X.numel() == 0:
        return 0.0

    X = X.to(device)
    Y = Y.to(device)

    correct = 0
    total = Y.size(0)

    batch_size = 64

    with torch.no_grad():
        for i in range(0, total, batch_size):
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]

            logits, _, _ = model(xb)
            preds = logits.argmax(dim=-1)

            correct += (preds == yb).sum().item()

    acc = 100.0 * correct / total
    return acc


def load_dailydialog_instruct():
    path = "/kaggle/working/dailydialog/data/dialogues.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instruct_samples = []

    for dialog in data:
        turns = dialog["turns"]

        for i in range(len(turns) - 1):
            u1 = turns[i]
            u2 = turns[i + 1]

            if u1["speaker"] == "user" and u2["speaker"] == "system":
                instruct_samples.append({
                    "user": u1["utterance"].strip(),
                    "assistant": u2["utterance"].strip()
                })

    return instruct_samples


def download_dailydialog_turns():
    url = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip?download=true"
    zip_path = "dailydialog.zip"
    extract_dir = "dailydialog"

    
    if not os.path.exists(zip_path):
        print("Downloading DailyDialog ZIP…")
        r = requests.get(url)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)

    
    if not os.path.exists(extract_dir):
        print("Extracting DailyDialog ZIP…")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)


SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and politely."


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stream_wikipedia(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line



def stream_batches(path, tok, batch_size=16, max_workers=4, prefetch=32):
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    def submit(line):
        return executor.submit(tok.encode, line)

    
    for line in stream_wikipedia(path):
        futures.append(submit(line))
        if len(futures) >= prefetch:
            break

    while True:
        batch_ids = []

        
        for _ in range(batch_size):
            if not futures:
                break
            future = futures.pop(0)
            ids = future.result()
            batch_ids.append(ids)

        if not batch_ids:
            break

        
        for line in stream_wikipedia(path):
            futures.append(submit(line))
            if len(futures) >= prefetch:
                break

        
        max_len = max(len(x) for x in batch_ids)
        batch = torch.full((len(batch_ids), max_len), tok.pad_id, dtype=torch.long)

        for i, seq in enumerate(batch_ids):
            batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        yield batch


def prepare_batch(batch, device, vocab_size, pad_id=0):
    lens = [len(seq) for seq in batch]
    m = max(lens)

    x = torch.full((len(batch), m), fill_value=pad_id, dtype=torch.long)
    y = torch.full((len(batch), m), fill_value=pad_id, dtype=torch.long)

    for i, seq in enumerate(batch):
        
        seq = seq.clone()
        
        seq = torch.clamp(seq, min=0, max=vocab_size - 1)
        L = len(seq)

        x[i, :L] = seq

        if L == 1:
            
            y[i, 0] = seq[0]
        else:
            
            y_seq = seq.clone()
            y_seq[:-1] = seq[1:]
            y_seq[-1] = seq[-1]
            y[i, :L] = y_seq

    return x.to(device), y.to(device)


def eval_stream_loss(model, tok, path, device,
                     batch_size=64, max_batches=50, pad_id=0):
    model.eval()
    total_loss = 0.0
    count = 0
    vocab_size = model.lm_head.weight.size(0)

    with torch.no_grad():
        for batch in stream_batches(path, tok, batch_size):
            x, y = prepare_batch(batch, device, vocab_size, pad_id=pad_id)
            logits = model(x)  
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=pad_id,
            )
            total_loss += loss.item()
            count += 1
            if count >= max_batches:
                break

    if count == 0:
        return 0.0
    return total_loss / count

def train_step(real_model, mirror_model, hippocampus, tok,
               batch, device, opt, scaler, opt_mirror, pad_id=0):

    vocab_size = real_model.lm_head.weight.size(0)
    x, y = prepare_batch(batch, device, vocab_size, pad_id=pad_id)

    
    
    
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits_real = real_model(x)
        loss_real = F.cross_entropy(
            logits_real.view(-1, vocab_size),
            y.view(-1),
            ignore_index=pad_id,
            label_smoothing=0.1,
        )

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss_real).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    
    
    
    
    logits_mirror, h_mirror, pred_error = mirror_model(x)

    
    y_last = y[:, -1]
    y_last = torch.clamp(y_last, 0, vocab_size - 1)

    loss_mirror = F.cross_entropy(logits_mirror, y_last)

    opt_mirror.zero_grad()
    loss_mirror.backward()
    opt_mirror.step()

    
    
    
    with torch.no_grad():
        probs = torch.softmax(logits_mirror, dim=-1)
        idx = torch.arange(y_last.size(0), device=device)
        conf = probs[idx, y_last].mean().item()
        surprise = 1.0 - conf

    
    sent_ids = batch[0].tolist()
    sentence = tok.decode(sent_ids)
    hippocampus.store(sentence, float(surprise * surprise))

    return float(loss_real.item()), float(pred_error), float(surprise)

def count_chunk(chunk):
    return Counter(chunk.split())



def observe_stream(tok, path, lines_per_chunk=100_000):
    total = Counter()
    buf = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                buf.append(line)

            if len(buf) >= lines_per_chunk:
                df = cudf.DataFrame({"text": buf})
                tokens = df["text"].str.split()
                flat = tokens.explode()
                counts = flat.value_counts()
                total.update(dict(counts.to_pandas()))
                buf = []

    if buf:
        df = cudf.DataFrame({"text": buf})
        tokens = df["text"].str.split()
        flat = tokens.explode()
        counts = flat.value_counts()
        total.update(dict(counts.to_pandas()))

    tok.vocab = total


def main():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        device_real = torch.device("cuda:0")
        device_mirror = torch.device("cuda:1")
    elif torch.cuda.is_available():
        device_real = torch.device("cuda:0")
        device_mirror = torch.device("cuda:0")
    else:
        device_real = torch.device("cpu")
        device_mirror = torch.device("cpu")

    tok = SPTokenizer("/kaggle/input/models/joepvanopdorp/tokenizer/jax/default/1/tokenizer.model")

    vocab_size = tok.vocab_size_actual

    size = 1024
    context = 128
    lr = 3e-5
    batch_size = 16
    max_epochs = 3

    real_model = LiquidLM(vocab_size, size, context).to(device_real)
    mirror_model = MirrorLM(vocab_size, size, size, context).to(device_mirror)

    hippocampus = Hippocampus(max_episodes=size).to(device_real)

    opt = optim.AdamW(real_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    opt_mirror = optim.AdamW(mirror_model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    scaler = torch.amp.GradScaler("cuda" if device_real.type == "cuda" else "cpu")


    print("parameters:", count_params(real_model))
    stop=True
    last = perf_counter()
    for epoch in range(max_epochs):
        real_model.train()
        mirror_model.train()
        for batch_i, batch in enumerate(stream_batches("/kaggle/input/datasets/joepvanopdorp/wikipedia-en-download/train.txt", tok, batch_size)):
            loss_r, pe, surpr = train_step(real_model, mirror_model, hippocampus, tok, batch, device_real, opt,scaler,opt_mirror)
            if batch_i % 10 == 0:
                print(f"epoch {epoch+1}/{max_epochs} | batch {batch_i} | train_loss={loss_r:.4f} | pred_err={pe:.4f} | surpr={surpr:.4f} | time {perf_counter()-last:.4f}")
                last = perf_counter()
            if batch_i >= 20_000:
                stop = True
                break
        if stop == True:
            break
        scheduler.step()
        test_loss = eval_stream_loss(real_model, tok, "/kaggle/input/datasets/joepvanopdorp/wikipedia-en-download/train.txt", device_real, batch_size=64, max_batches=50)
        print(f"epoch {epoch+1} | test_loss={test_loss:.4f}")

    torch.save({"model": real_model.state_dict(),"mirror_model":mirror_model.state_dict()}, "model.pth")


if __name__ == "__main__":
    main()
