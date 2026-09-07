import torch, numpy as np
torch.manual_seed(7)
H, T, L = 3, 5, 2
lstm = torch.nn.LSTM(H, H, L)
x = torch.randn(T, 1, H)
with torch.no_grad():
    y, _ = lstm(x)
out = (y + x)[:, 0].numpy()

def arr(name, a):
    vals = np.asarray(a, np.float32).ravel()
    lines, cur = [], "        "
    for i, v in enumerate(vals):
        piece = f"{v:.9g}f" + ("," if i + 1 < len(vals) else "")
        if len(cur) + len(piece) > 92:
            lines.append(cur.rstrip()); cur = "        "
        cur += piece + " "
    lines.append(cur.rstrip())
    return f"    static const float {name}[] = {{\n" + "\n".join(lines) + "\n    };"

for l in range(L):
    print(arr(f"w_ih{l}", getattr(lstm, f"weight_ih_l{l}").detach()))
    print(arr(f"w_hh{l}", getattr(lstm, f"weight_hh_l{l}").detach()))
    print(arr(f"bias{l}", getattr(lstm, f"bias_ih_l{l}").detach()
                          + getattr(lstm, f"bias_hh_l{l}").detach()))
print(arr("input_seq", x[:, 0]))
print(arr("expected", out))
