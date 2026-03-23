import torch
import random
import model as Cmodel
import torch.nn.functional as F


def micro_train_eight_bits(model, tok, steps=800, seq_len=256, device="cpu"):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    A_id = tok.encode("A")[0]
    B_id = tok.encode("B")[0]

    for step in range(steps):
        bits = [random.choice([A_id, B_id]) for _ in range(8)]
        filler = [random.choice([A_id, B_id]) for _ in range(seq_len - 8)]
        seq = bits + filler

        losses = []
        for j in range(8):
            rotated = seq[j+1:] + seq[:j+1]

            x = torch.tensor([rotated], dtype=torch.long, device=device)
            target = torch.tensor([bits[j]], dtype=torch.long, device=device)

            logits, _,_  = model(x)
            loss_j = F.cross_entropy(logits, target)
            losses.append(loss_j)

        loss = sum(losses)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[train-8bit] step {step}, loss={loss.item():.4f}")

    print("Eight-bit micro-training complete.\n")


def test_eight_bit_context_window(model, tok, device="cpu"):

    print("Testing 8-bit effective context window...")
    model.eval()

    A_id = tok.encode("A")[0]
    B_id = tok.encode("B")[0]

    max_len = 0
    step = 256
    seq_len = step

    while True:
        bits = [random.choice([A_id, B_id]) for _ in range(8)]
        filler = [random.choice([A_id, B_id]) for _ in range(seq_len - 8)]
        seq = bits + filler

        all_ok = True

        for j in range(8):
            rotated = seq[j+1:] + seq[:j+1]

            x = torch.tensor([rotated], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ ,_= model(x)
                pred = logits.argmax(dim=-1).item()

            if pred != bits[j]:
                all_ok = False
                print(f"FAIL at distance {seq_len} on bit {j}")
                print(f"pred={pred}, true={bits[j]}")
                break

        if all_ok:
            print(f"OK: remembers ALL 8 bits at distance {seq_len}")
            max_len = seq_len
            seq_len += step
        else:
            break

    print("\n=== RESULT ===")
    print("8-bit effective context window:", max_len)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


    tok = Cmodel.DynamicTokenizer()
    tok.build_char_vocab(["A", "B"])


    vocab_size = tok.vocab_size_actual
    model = Cmodel.LiquidLM(vocab_size=vocab_size, d_model=64, hidden_size=64)
    model.to(device)

    model.rel_world.store = lambda *args, **kwargs: None
    model.rel_world.query = lambda *args, **kwargs: torch.zeros(
        (1, model.hidden_size), device=device
    )


    micro_train_eight_bits(model, tok, steps=800, seq_len=256, device=device)

    test_eight_bit_context_window(model, tok, device=device)


if __name__ == "__main__":
    main()
