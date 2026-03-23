import re
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import multiprocessing as mp
from collections import Counter
from torch.amp import autocast, GradScaler


def export_model_state(tok, real_model, hippocampus, filename="model_export.json"):
    export_data = {}

    vocab_list = [tok.id2token[i] for i in range(len(tok.id2token))]
    export_data["vocabulary"] = vocab_list


    wm = real_model.wm.init_content.detach().cpu().tolist()
    export_data["working_memory"] = wm

    export_data["hippocampus"] = [
        {"sentence": s, "priority": p}
        for (s, p) in hippocampus.episodes
    ]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Model state exported to {filename}")


class DynamicTokenizer:
    def __init__(self, min_subword_freq=10, min_word_freq=3):
        self.special_tokens = ["<pad>", "<unk>"]
        self.token2id = {t: i for i, t in enumerate(self.special_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.char2id = {}
        self.subword2id = {}
        self.word2id = {}
        self.word_freq = {}
        self.subword_freq = {}
        self.min_subword_freq = min_subword_freq
        self.min_word_freq = min_word_freq

    def _add_token(self, tok):
        if tok not in self.token2id:
            idx = len(self.token2id)
            self.token2id[tok] = idx
            self.id2token[idx] = tok

    @property
    def pad_id(self):
        return self.token2id["<pad>"]

    @property
    def vocab_size_actual(self):
        return len(self.token2id)

    def build_char_vocab(self, corpus):
        chars = set()
        for line in corpus:
            for ch in line:
                if not ch.isspace():
                    chars.add(ch)
        for ch in sorted(chars):
            tok = f"<ch:{ch}>"
            self._add_token(tok)
            self.char2id[ch] = self.token2id[tok]

    def _count_pairs_chunk(self, chunk):
        c = Counter()
        for w in chunk:
            for i in range(len(w) - 1):
                c[(w[i], w[i+1])] += 1
        return c

    def train_subwords(self, corpus, max_merges=5000, n_workers=4):
        for ch in self.char2id:
            self.subword2id[ch] = self.char2id[ch]

        def split_word(w):
            return list(w)

        vocab = []
        for line in corpus:
            for w in line.strip().split():
                vocab.append(split_word(w))

        if not vocab:
            return

        for _ in range(max_merges):
            if len(vocab) < n_workers:
                chunks = [vocab]
            else:
                chunk_size = max(1, len(vocab) // n_workers)
                chunks = [vocab[i:i+chunk_size] for i in range(0, len(vocab), chunk_size)]

            with mp.Pool(n_workers) as pool:
                counters = pool.map(self._count_pairs_chunk, chunks)

            pair_counts = Counter()
            for c in counters:
                pair_counts.update(c)

            if not pair_counts:
                break

            (a, b), freq = pair_counts.most_common(1)[0]
            if freq < self.min_subword_freq:
                break

            new_sw = a + b
            if new_sw in self.subword2id:
                continue

            tok = f"<sw:{new_sw}>"
            self._add_token(tok)
            self.subword2id[new_sw] = self.token2id[tok]

            new_vocab = []
            for w in vocab:
                i = 0
                merged = []
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i+1] == b:
                        merged.append(new_sw)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                new_vocab.append(merged)
            vocab = new_vocab

    def observe_sentence(self, sentence):
        words = sentence.strip().split()
        new_tokens = []
        for w in words:
            self.word_freq[w] = self.word_freq.get(w, 0) + 1
            if self.word_freq[w] >= self.min_word_freq and w not in self.word2id:
                tok = f"<w:{w}>"
                self._add_token(tok)
                self.word2id[w] = self.token2id[tok]
                new_tokens.append(tok)
        return new_tokens

    def _encode_word(self, w):
        if w in self.word2id:
            return [self.word2id[w]]
        ids = []
        i = 0
        L = len(w)
        while i < L:
            best = None
            best_len = 0
            for l in range(1, L - i + 1):
                sw = w[i:i+l]
                if sw in self.subword2id:
                    best = self.subword2id[sw]
                    best_len = l
            if best is not None:
                ids.append(best)
                i += best_len
            else:
                ch = w[i]
                if ch in self.char2id:
                    ids.append(self.char2id[ch])
                else:
                    ids.append(self.token2id["<unk>"])
                i += 1
        return ids

    def encode(self, text):
        ids = []
        for w in text.strip().split():
            ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        pieces = []
        for i in ids:
            if i not in self.id2token:
                continue
            tok = self.id2token[i]
            if tok in self.special_tokens:
                continue
            if tok.startswith("<w:"):
                pieces.append(tok[3:-1])
            elif tok.startswith("<sw:"):
                pieces.append(tok[4:-1])
            elif tok.startswith("<ch:"):
                pieces.append(tok[4:-1])
        return "".join(pieces)

def build_sequences_sp(corpus, tok, window=16):
    X = []
    Y = []
    pad = tok.pad_id
    for line in corpus:
        ids = tok.encode(line)
        if len(ids) < 2:
            continue
        for i in range(1, len(ids)):
            ctx = ids[max(0, i - window):i]
            ctx = [pad] * (window - len(ctx)) + ctx
            nxt = ids[i]
            X.append(ctx)
            Y.append(nxt)
    if not X:
        return torch.empty(0, window, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

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
    def __init__(self, hidden_size, max_nodes=1024, max_edges=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.register_buffer("node_reprs", torch.empty(0, hidden_size))
        self.node_ids = []

        self.register_buffer("edge_src",  torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_dst",  torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_rel",  torch.empty(0, hidden_size))
        self.register_buffer("edge_conf", torch.empty(0, 1))

        self.W_msg = nn.Linear(hidden_size * 2, hidden_size)
        self.W_self = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()

    def _device(self):
        return self.W_msg.weight.device

    def _add_or_get_node(self, vec, node_id=None, sim_thresh=0.9):
        with torch.no_grad():
            v = vec.mean(dim=0, keepdim=True).detach().to(self._device())
            if self.node_reprs.numel() == 0:
                self.node_reprs = v
                self.node_ids = [node_id]
                return 0

            v_norm = F.normalize(v, dim=-1)
            nodes_norm = F.normalize(self.node_reprs, dim=-1)
            sims = (v_norm @ nodes_norm.t()).squeeze(0)
            best_i = int(sims.argmax().item())
            if float(sims[best_i]) >= sim_thresh:
                return best_i

            self.node_reprs = torch.cat([self.node_reprs, v], dim=0)
            self.node_ids.append(node_id)

            if self.node_reprs.size(0) > self.max_nodes:
                excess = self.node_reprs.size(0) - self.max_nodes
                self.node_reprs = self.node_reprs[excess:]
                self.node_ids = self.node_ids[excess:]
                self.edge_src  = self.edge_src.new_empty(0)
                self.edge_dst  = self.edge_dst.new_empty(0)
                self.edge_rel  = self.edge_rel.new_empty(0, self.hidden_size)
                self.edge_conf = self.edge_conf.new_empty(0, 1)

            return self.node_reprs.size(0) - 1

    def store(self, subj_vec, rel_vec, obj_vec, confidence=1.0):
        with torch.no_grad():
            device = self._device()
            s_idx = self._add_or_get_node(subj_vec)
            o_idx = self._add_or_get_node(obj_vec)

            rel = rel_vec.mean(dim=0, keepdim=True).detach().to(device)
            conf = torch.tensor([[float(confidence)]], device=device)

            src = torch.tensor([s_idx], dtype=torch.long, device=device)
            dst = torch.tensor([o_idx], dtype=torch.long, device=device)

            if self.edge_src.numel() == 0:
                self.edge_src  = src
                self.edge_dst  = dst
                self.edge_rel  = rel
                self.edge_conf = conf
            else:
                self.edge_src  = torch.cat([self.edge_src,  src], dim=0)
                self.edge_dst  = torch.cat([self.edge_dst,  dst], dim=0)
                self.edge_rel  = torch.cat([self.edge_rel,  rel], dim=0)
                self.edge_conf = torch.cat([self.edge_conf, conf], dim=0)

            if self.edge_src.size(0) > self.max_edges:
                excess = self.edge_src.size(0) - self.max_edges
                self.edge_src  = self.edge_src[excess:]
                self.edge_dst  = self.edge_dst[excess:]
                self.edge_rel  = self.edge_rel[excess:]
                self.edge_conf = self.edge_conf[excess:]

    def _message_passing(self, node_states, steps=2):
        if self.edge_src.numel() == 0:
            return node_states

        device = node_states.device
        src  = self.edge_src.to(device)
        dst  = self.edge_dst.to(device)
        rel  = self.edge_rel.to(device)
        conf = self.edge_conf.to(device)

        for _ in range(steps):
            src_h = node_states[src]
            msg_in = torch.cat([src_h, rel], dim=-1)
            msg = self.act(self.W_msg(msg_in)) * conf
            agg_msgs = torch.zeros_like(node_states)
            agg_msgs.index_add_(0, dst, msg)
            node_states = self.act(self.W_self(node_states) + agg_msgs)

        return node_states

    def query(self, entity_vec, k=4):
        if self.node_reprs.numel() == 0:
            return torch.zeros_like(entity_vec.mean(dim=-2))

        with torch.no_grad():
            device = entity_vec.device
            node_states = self.node_reprs.to(device)
            node_states = self._message_passing(node_states, steps=2)

            v = entity_vec.mean(dim=0, keepdim=True)
            v_norm = F.normalize(v, dim=-1)
            nodes_norm = F.normalize(node_states, dim=-1)
            sims = (v_norm @ nodes_norm.t()).squeeze(0)
            best_i = int(sims.argmax().item())
            ctx = node_states[best_i:best_i+1]
            B = entity_vec.size(0)
            return ctx.expand(B, -1)


class RelationalGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, concept_mix):
        return torch.sigmoid(self.gate(concept_mix))

class ConceptExtractor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.subj_proj = nn.Linear(hidden_size, hidden_size)
        self.act_proj  = nn.Linear(hidden_size, hidden_size)
        self.obj_proj  = nn.Linear(hidden_size, hidden_size)
        self.subj_score = nn.Linear(hidden_size, 1)
        self.act_score  = nn.Linear(hidden_size, 1)
        self.obj_score  = nn.Linear(hidden_size, 1)

    def forward(self, token_states):
        B, T, H = token_states.shape
        subj_w = torch.softmax(self.subj_score(token_states).squeeze(-1), dim=-1)
        act_w  = torch.softmax(self.act_score(token_states).squeeze(-1), dim=-1)
        obj_w  = torch.softmax(self.obj_score(token_states).squeeze(-1), dim=-1)

        subj_vec = torch.bmm(subj_w.unsqueeze(1), self.subj_proj(token_states)).squeeze(1)
        act_vec  = torch.bmm(act_w.unsqueeze(1),  self.act_proj(token_states)).squeeze(1)
        obj_vec  = torch.bmm(obj_w.unsqueeze(1),  self.obj_proj(token_states)).squeeze(1)
        return subj_vec, act_vec, obj_vec

class ConfidenceNet(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size + vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act = nn.ReLU()

    def forward(self, h, logits):
        probs = torch.softmax(logits, dim=-1)
        x = torch.cat([h, probs], dim=-1)
        x = self.act(self.fc1(x))
        conf = torch.sigmoid(self.fc2(x))
        return conf


class WorkingMemory(nn.Module):
    def __init__(self, num_slots=16, slot_dim=64):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.init_content = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.01)
        self.write_proj = nn.Linear(slot_dim, slot_dim)
        self.write_gate = nn.Linear(slot_dim, 1)
        self.addr_proj = nn.Linear(slot_dim, num_slots)

    def init_state(self, batch_size, device=None):
        if device is None:
            device = self.init_content.device
        return self.init_content.unsqueeze(0).expand(batch_size, self.num_slots, self.slot_dim).to(device)

    def forward(self, query, wm_state=None):
        if wm_state is not None:
            wm_state = wm_state.detach()

        B, D = query.shape
        S = self.num_slots
        device = query.device
        if wm_state is None:
            wm_state = self.init_state(B, device=device)

        attn_logits = torch.bmm(wm_state, query.unsqueeze(-1)).squeeze(-1)
        attn = F.softmax(attn_logits, dim=-1)
        read_vec = torch.bmm(attn.unsqueeze(1), wm_state).squeeze(1)

        addr_logits = self.addr_proj(query)
        addr = F.softmax(addr_logits, dim=-1)
        gate = torch.sigmoid(self.write_gate(query))
        write_content = torch.tanh(self.write_proj(query))

        addr_exp = addr.unsqueeze(-1)
        write_vec = write_content.unsqueeze(1)
        gate_exp = gate.unsqueeze(-1)
        delta = addr_exp * write_vec * gate_exp
        new_wm_state = wm_state + delta
        return read_vec, new_wm_state

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
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        q = self.phi(q)
        k = self.phi(k)

        k_sum = k.sum(dim=1)
        kv_sum = torch.einsum("bthd,bthd->bhd", k, v)
        kv_sum = kv_sum.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)

        out = torch.einsum("bthd,bhdv->bthv", q, kv_sum)
        z = torch.einsum("bthd,bhd->bth", q, k_sum)
        z = (z + self.eps).unsqueeze(-1)

        out = out / z
        out = out.reshape(B, T, D)

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
        device = x.device
        dtype = x.dtype

        dt = torch.exp(self.log_dt)
        lambda_ = -torch.exp(self.log_lambda)      
        decay = torch.exp(lambda_ * dt)              

        h = torch.zeros(B, self.d_state, device=device, dtype=dtype)
        outputs = []

        for t in range(T):
            u = x[:, t, :]                        
            Bu = u @ self.B.T                      
            h = h * decay.unsqueeze(0) + Bu         
            y = h @ self.C.T                         
            outputs.append(y.unsqueeze(1))

        return torch.cat(outputs, dim=1)    

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=128):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.ssm = S4DSSM(d_state=d_state, d_model=d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_in = self.in_proj(x)
        x_ssm = self.ssm(x_in)
        x_out = x_in + x_ssm
        x_out = self.out_proj(x_out)
        x_out = self.norm(x_out)
        return x_out


class LiquidLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_size=64, window=16, num_layers=3):
        super().__init__()
        assert d_model <= 768
        self.window = window
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._wm_state = None

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = LiquidSelfAttention(d_model, num_heads=2)
        self.pre_norm = nn.LayerNorm(d_model)
        self.mamba_layers = nn.ModuleList([MambaBlock(d_model, d_state=128) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        self.wm = WorkingMemory(num_slots=16, slot_dim=hidden_size)
        self.wm_proj = nn.Linear(hidden_size, hidden_size)

        self.concepts = ConceptExtractor(hidden_size)
        self.rel_world = RelationalWorldModel(hidden_size)
        self.rel_gate = RelationalGate(hidden_size)
        self.rel_proj = nn.Linear(hidden_size, hidden_size)

        self.conf_net = ConfidenceNet(hidden_size, vocab_size)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, hidden_size)
        )
        self.adapter.requires_grad_(False)

        self.proj_to_hidden = nn.Linear(d_model, hidden_size)

    def grow_vocab(self, new_vocab_size, tok):
        old_vocab = self.lm_head.weight.data.shape[0]
        if new_vocab_size <= old_vocab:
            return

        device = self.embedding.weight.device
        self.embedding = grow_embedding(self.embedding, new_vocab_size).to(device)

        old_w = self.lm_head.weight.data
        old_b = self.lm_head.bias.data
        old_vocab, dim = old_w.shape
        new_head = nn.Linear(dim, new_vocab_size).to(device)
        new_head.weight.data[:old_vocab] = old_w.clone()
        new_head.bias.data[:old_vocab] = old_b.clone()
        nn.init.normal_(new_head.weight.data[old_vocab:], mean=0.0, std=0.02)
        nn.init.zeros_(new_head.bias.data[old_vocab:])
        self.lm_head = new_head

        self.conf_net = ConfidenceNet(self.hidden_size, new_vocab_size).to(device)
        tok.token2id = {tok.id2token[i]: i for i in range(len(tok.id2token))}

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)
        x = self.attn(x)
        x = self.pre_norm(x)

        for layer in self.mamba_layers:
            x = layer(x)

        x_hidden = self.proj_to_hidden(x)
        gru_out = x_hidden
        h_last = gru_out[:, -1, :]

        subj_vec, act_vec, obj_vec = self.concepts(gru_out)
        rel_ctx = self.rel_world.query(subj_vec)
        concept_mix = subj_vec + act_vec + obj_vec + rel_ctx
        concept_mix = self.rel_proj(concept_mix)
        gate = self.rel_gate(concept_mix)
        h_last = h_last + gate * concept_mix

        h_last = h_last + self.adapter(h_last)

        query = self.wm_proj(h_last)
        if self._wm_state is None or self._wm_state.size(0) != B:
            self._wm_state = self.wm.init_state(B, device=device)
        self._wm_state = self._wm_state.detach().clone()
        wm_read, self._wm_state = self.wm(query, wm_state=self._wm_state)
        h_last = h_last + wm_read

        s4d_summary = x_hidden[:, -1, :]
        h_final = h_last + s4d_summary

        h_final = self.ln(h_final)
        h_final = self.dropout(h_final)
        logits = self.lm_head(h_final)
        conf = self.conf_net(h_final.detach(), logits.detach())

        return logits, conf, h_final



class MirrorLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_size=64, window=16):
        super().__init__()
        self.window = window
        self.hidden_size = hidden_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = LiquidSelfAttention(d_model, num_heads=2)
        self.liquid = LiquidGRUCell(d_model, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        self.register_buffer("fast_W", torch.zeros(hidden_size, hidden_size))
        self.meta_rnn = nn.GRUCell(2, 16)
        self.meta_out_lr = nn.Linear(16, 1)
        self.meta_out_fw = nn.Linear(16, 1)
        self.meta_state = None
        self.vocab_size = vocab_size

    def grow_vocab(self, new_vocab_size):
        old_emb = self.embedding.weight.data
        old_vocab, dim = old_emb.shape
        if new_vocab_size <= old_vocab:
            return
        device = old_emb.device
        new_emb = nn.Embedding(new_vocab_size, dim).to(device)
        new_emb.weight.data[:old_vocab] = old_emb.clone()
        nn.init.normal_(new_emb.weight.data[old_vocab:], mean=0.0, std=0.02)
        self.embedding = new_emb

        old_w = self.lm_head.weight.data
        old_b = self.lm_head.bias.data
        new_head = nn.Linear(self.hidden_size, new_vocab_size).to(device)
        new_head.weight.data[:old_vocab] = old_w.clone()
        new_head.bias.data[:old_vocab] = old_b.clone()
        nn.init.normal_(new_head.weight.data[old_vocab:], mean=0.0, std=0.02)
        nn.init.zeros_(new_head.bias.data[old_vocab:])
        self.lm_head = new_head

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        x = self.attn(x)

        h = torch.zeros(B, self.hidden_size, device=input_ids.device)
        prev_h = None
        last_pred_error = torch.zeros(B, self.hidden_size, device=input_ids.device)

        for t in range(T):
            slow_h = self.liquid(x[:, t, :], h)
            fast_term = h @ self.fast_W
            h_next = slow_h + fast_term
            if prev_h is not None:
                pred_error = h_next - prev_h
                last_pred_error = pred_error
            prev_h = h_next
            h = h_next

        h = self.ln(h)
        h = self.dropout(h)
        logits = self.lm_head(h)
        pred_error_norm = last_pred_error.pow(2).mean().sqrt().item()
        return logits, h, pred_error_norm

    def meta_step(self, loss_val, surprise, device="cpu"):
        x = torch.tensor([[loss_val, surprise]], dtype=torch.float32, device=device)
        if self.meta_state is None:
            self.meta_state = torch.zeros(1, 16, device=device)
        self.meta_state = self.meta_rnn(x, self.meta_state)
        lr = torch.sigmoid(self.meta_out_lr(self.meta_state)).item() * 0.001
        fw_scale = torch.sigmoid(self.meta_out_fw(self.meta_state)).item() * 0.1

        return lr, fw_scale

def train_epoch(real_model, tok, corpus, device, window=16, scheduler=None, opt=None, lr=1e-4):
    real_model.train()
    X, Y = build_sequences_sp(corpus, tok, window=window)
    if X.numel() == 0:
        return 0.0, opt

    X = X.to(device)
    Y = Y.to(device)

    if opt is None:
        opt = optim.AdamW(real_model.parameters(), lr=lr)

    scaler = GradScaler(device=device)

    batch_size = 256
    total_loss = 0.0
    n_batches = 0

    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size]
        yb = Y[i:i+batch_size]

        opt.zero_grad()

        with autocast("cuda"):
            logits, _, _ = real_model(xb)
            loss = F.cross_entropy(logits, yb)

        scaler.scale(loss).backward()


        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)

        scaler.step(opt)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches), opt


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

    new_tokens = tok.observe_sentence(sentence)
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
def reset_rel_world(model):
    rw = model.rel_world
    rw.node_reprs = torch.empty(0, model.hidden_size, device=rw.W_msg.weight.device)
    rw.edge_src  = torch.empty(0, dtype=torch.long, device=rw.W_msg.weight.device)
    rw.edge_dst  = torch.empty(0, dtype=torch.long, device=rw.W_msg.weight.device)
    rw.edge_rel  = torch.empty(0, model.hidden_size, device=rw.W_msg.weight.device)
    rw.edge_conf = torch.empty(0, 1, device=rw.W_msg.weight.device)

def compute_accuracy(model, X, Y, device):
    model.eval()

    if X.numel() == 0:
        return 0.0

    X = X.to(device)
    Y = Y.to(device)

    correct = 0
    total = Y.size(0)

    batch_size = 256

    with torch.no_grad():
        for i in range(0, total, batch_size):
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]

            logits, _, _ = model(xb)
            preds = logits.argmax(dim=-1)

            correct += (preds == yb).sum().item()

    acc = 100.0 * correct / total
    return acc

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

    import json
    with open("/kaggle/input/datasets/joepvanopdorp/curriculum-english2/curriculum.json", "r", encoding="utf-8") as f:
        curriculum = json.load(f)

    phases = [
        ("phase1_simple_svo", curriculum["phase1_simple_svo"], 19),
        ("phase2_svo_adv", curriculum["phase2_svo_adv"], 13.3),
        ("phase3_svo_prep_loc", curriculum["phase3_svo_prep_loc"],323),
        ("phase4_compound", curriculum["phase4_compound"], 13.7),
        ("phase5_stories", curriculum["phase5_stories"], 16),
    ]

    tok = DynamicTokenizer()
    vocab_size = 120
    size = 512
    context = 128
    lr = 5e-4
    real_model = LiquidLM(vocab_size, size, size, context).to(device_real)
    mirror_model = MirrorLM(vocab_size, size, size, context).to(device_mirror)
    hippocampus = Hippocampus(max_episodes=size).to(device_real)
    opt = optim.AdamW(real_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)

    max_epochs = 40
    window = 8

    def test_generation(model, tok, prompt, device):
        ids = tok.encode(prompt)
        if len(ids) < 1:
            return ""
        ctx = ids[-16:]
        ctx = [tok.pad_id] * (16 - len(ctx)) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _, _ = model(x)
        nid = int(logits.argmax(dim=-1).item())
        return tok.decode([nid])

    def test_perplexity(model, tok, data, device):
        seqs = build_sequences_sp(data, tok)
        X, Y = seqs
        with torch.no_grad():
            logits, _, _ = model(X.to(device))
            loss = torch.nn.functional.cross_entropy(logits, Y.to(device))
        return float(torch.exp(loss).item())


    for phase_name, phase_data, phase_ppl in phases:
        print(f"\n=== Training {phase_name} ===")
        new_tokens = tok.observe_sentence(" ".join(phase_data))
        if new_tokens:
            new_vocab_size = tok.vocab_size_actual
            real_model.grow_vocab(new_vocab_size, tok)
            mirror_model.grow_vocab(new_vocab_size)
            real_model.to(device_real)
            mirror_model.to(device_mirror)

        for epoch in range(max_epochs):
            loss = train_epoch(real_model, tok, phase_data, device_real, window=window,scheduler=scheduler,lr=lr,opt=opt)
            acc = compute_accuracy(real_model, *build_sequences_sp(phase_data, tok), device_real)
            ppl = test_perplexity(real_model, tok, phase_data, device_real)
            print(f"{phase_name} | epoch {epoch+1}/{max_epochs} | loss={loss[0]:.2f} | acc={acc:.2f}% | ppl={ppl:.2f}")
            if ppl < phase_ppl:
                print(f"{phase_name} mastered")
                break
            reset_rel_world(real_model)
            real_model._wm_state = None

        sample_prompt = phase_data[0].split()[0] + " "
        gen = test_generation(real_model, tok, sample_prompt, device_real)
        print("sample generation:", sample_prompt, "->", gen)

 


    real_model.eval()
    prompt = "The baby eats the "
    out = test_generation(real_model, tok, prompt, device_real)
    print(f"\nFinal test: '{prompt}' → '{out}'")

    export_model_state(tok, real_model, hippocampus, filename="model_export.json")
    torch.save(real_model.state_dict(), "model.pth")
    print("Model saved as model.pth")

if __name__ == "__main__":
    main()

 
