import torch
import torch.nn.functional as F
from model_train import LiquidLM, DynamicTokenizer
import torch.serialization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and politely."


torch.serialization.add_safe_globals([DynamicTokenizer])


def load_checkpoint(path="model.pth"):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    tok = checkpoint["tokenizer"]
    model_state = checkpoint["model"]
    return tok, model_state


def load_model(tok, model_state):
    model = LiquidLM(
        vocab_size=len(tok.token2id),
        d_model=512,
        hidden_size=512,
        window=64,
        num_layers=4,
        use_checkpoint=False
    )
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()
    return model



def sample_next_token(logits, temperature=0.8, top_k=40, repetition_penalty=1.1, recent_ids=None):
    logits = logits.clone()

    if recent_ids:
        for tid in recent_ids:
            logits[tid] /= repetition_penalty

    logits = logits / temperature

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits[logits < min_val] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()



def generate(model, tok, prompt, max_new_tokens=30, seq_len=128):
    ids = tok.encode(prompt)
    ids = ids[-seq_len:]

    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

        next_id = sample_next_token(
            logits[0, -1],
            temperature=0.2,    
            top_k=30,           
            repetition_penalty=1.2,
            recent_ids=ids[-25:]
        )

        ids.append(next_id)
        ids = ids[-seq_len:]

    gen = ids[len(tok.encode(prompt)):]
    return tok.decode(gen)

def chat():
    tok, model_state = load_checkpoint()
    model = load_model(tok, model_state)

    print("Chatbot geladen. Typ 'exit' om te stoppen.\n")

    while True:
        user = input("Jij: ").strip()
        if user.lower() in ["exit", "quit", "stop"]:
            break

        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{user}\n"
            f"<|assistant|>\n"
        )

        reply = generate(model, tok, full_prompt)
        print("Bot:", reply)


if __name__ == "__main__":
    chat()
