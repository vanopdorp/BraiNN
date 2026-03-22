import torch
import random
import model
import torch.nn.functional as F

def micro_train_two_bits(model, tok, steps=800, seq_len=64, device="cpu"):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    A_id = tok.encode("A")[0]
    B_id = tok.encode("B")[0]

    for step in range(steps):
        first = random.choice([A_id, B_id])
        seq_mid = [random.choice([A_id, B_id]) for _ in range(seq_len - 2)]
        last = random.choice([A_id, B_id])
        seq = [first] + seq_mid + [last]

        x = torch.tensor([seq], dtype=torch.long, device=device)
        x_rev = torch.tensor([list(reversed(seq))], dtype=torch.long, device=device)

        opt.zero_grad()

        logits_last, _ = model(x)
        loss_last = F.cross_entropy(logits_last, torch.tensor([last], device=device))

        logits_first, _ = model(x_rev)
        loss_first = F.cross_entropy(logits_first, torch.tensor([first], device=device))

        loss = loss_first + loss_last
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"[train] step {step}, loss={loss.item():.4f}")

    print("Two-bit micro-training complete.\n")


def test_two_bit_context_window(model, tok, device="cpu"):
    print("Testing 2-bit effective context window...")
    model.eval()

    A_id = tok.encode("A")[0]
    B_id = tok.encode("B")[0]

    max_len = 0
    step = 64
    seq_len = step

    while True:
        first = random.choice([A_id, B_id])
        seq_mid = [random.choice([A_id, B_id]) for _ in range(seq_len - 2)]
        last = random.choice([A_id, B_id])
        seq = [first] + seq_mid + [last]

        x = torch.tensor([seq], dtype=torch.long, device=device)
        x_rev = torch.tensor([list(reversed(seq))], dtype=torch.long, device=device)

        with torch.no_grad():
            logits_last, _ = model(x)
            pred_last = logits_last.argmax(dim=-1).item()

            logits_first, _ = model(x_rev)
            pred_first = logits_first.argmax(dim=-1).item()

        ok_first = (pred_first == first)
        ok_last  = (pred_last == last)

        if ok_first and ok_last:
            print(f"OK: remembers FIRST+LAST at distance {seq_len}")
            max_len = seq_len
            seq_len += step
        else:
            print(f"FAIL: forgets at distance {seq_len}")
            print(f"pred_first={pred_first}, first={first}, pred_last={pred_last}, last={last}")
            break

    print("\n=== RESULT ===")
    print("2-bit effective context window:", max_len)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # tokenizer
    tok = v53.DynamicTokenizer()
    tok.build_char_vocab(["A", "B"])

    # model
    vocab_size = tok.vocab_size_actual
    model = v53.LiquidLM(vocab_size=vocab_size, d_model=64, hidden_size=64)
    model.to(device)

    # 1) train op 2 bits
    micro_train_two_bits(model, tok, steps=800, seq_len=64, device=device)

    # 2) test 2-bit context window
    test_two_bit_context_window(model, tok, device=device)


if __name__ == "__main__":
    main()

'''
FAIL: forgets at distance 33728
pred_first=2, first=2, pred_last=2, last=3

=== RESULT ===
2-bit effective context window: 33664
'''