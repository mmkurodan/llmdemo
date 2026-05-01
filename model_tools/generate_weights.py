#!/usr/bin/env python3
"""
Generate a tiny decoder-only Transformer whose weights match the Kotlin
MiniTransformer implementation used by the Android demo.

The exported JSON is intentionally simple:
1. config: model hyperparameters
2. embedding: token embedding table
3. layers: transposed linear weights plus LayerNorm parameters

Android reads the same JSON directly from app/src/main/assets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_vocab(path: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    vocab = json.loads(path.read_text(encoding="utf-8"))
    stoi = {token: index for index, token in enumerate(vocab)}
    itos = {index: token for index, token in enumerate(vocab)}
    return vocab, stoi, itos


def load_training_corpus(path: Path) -> list[str]:
    """Load UTF-8 text and skip blank lines so separators can be left in the file."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as corpus_file:
        for raw_line in corpus_file:
            line = raw_line.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError(f"Training corpus is empty: {path}")
    return lines


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """
    Convert a training string into token IDs.

    The corpus is character-level for normal text, but special tokens such as
    [BOS] and [EOS] should stay intact so the model can learn greeting patterns
    that start and stop explicitly.
    """
    unk_id = stoi["[UNK]"]
    multi_char_tokens = sorted((token for token in stoi if len(token) > 1), key=len, reverse=True)

    ids: list[int] = []
    cursor = 0
    while cursor < len(text):
        matched_token = None
        for token in multi_char_tokens:
            if text.startswith(token, cursor):
                matched_token = token
                break

        if matched_token is not None:
            ids.append(stoi[matched_token])
            cursor += len(matched_token)
            continue

        char = text[cursor]
        ids.append(stoi.get(char, stoi.get(char.lower(), unk_id)))
        cursor += 1

    return ids


def sinusoidal_positions(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_terms = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(length, d_model, dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(positions * div_terms)
    encoding[:, 1::2] = torch.cos(positions * div_terms)
    return encoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ff_hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, ff_hidden_dim, bias=False)
        self.ff2 = nn.Linear(ff_hidden_dim, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(x.shape[-1])
        causal_mask = torch.triu(
            torch.ones(scores.shape[-2:], dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, -1e9)
        attention = torch.softmax(scores, dim=-1)
        x = self.norm1(x + self.o(attention @ v))
        x = self.norm2(x + self.ff2(F.gelu(self.ff1(x))))
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        ff_hidden_dim: int,
        max_seq_len: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            DecoderLayer(d_model=d_model, ff_hidden_dim=ff_hidden_dim)
            for _ in range(num_layers)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device)
        for layer in self.layers:
            x = layer(x)
        return x @ self.embedding.weight.t()


def train_model(
    model: MiniTransformer,
    corpus_lines: list[str],
    stoi: dict[str, int],
    steps: int,
    lr: float,
) -> None:
    if steps <= 0:
        return

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(steps):
        # ランダムに別の行を引くことで、短いコーパスでも見える並びが毎回少し変わり、
        # 挨拶パターンを反復して学習しやすくする。
        line = random.choice(corpus_lines)
        ids = encode(line, stoi)
        if len(ids) < 2:
            raise ValueError(f"Training line must contain at least 2 tokens: {line!r}")

        # 既存のモデル長を超える行はランダム窓で切り出し、構造を変えずに学習対象を増やす。
        if len(ids) > model.max_seq_len + 1:
            start = random.randint(0, len(ids) - (model.max_seq_len + 1))
            ids = ids[start : start + model.max_seq_len + 1]

        x = ids[:-1]
        y = ids[1:]

        inputs = torch.tensor([x], dtype=torch.long)
        targets = torch.tensor([y], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")


def export_weights(
    model: MiniTransformer,
    vocab_size: int,
    d_model: int,
    ff_hidden_dim: int,
    max_seq_len: int,
    num_layers: int,
) -> dict:
    model.eval()

    exported_layers = []
    for layer in model.layers:
        exported_layers.append(
            {
                "w_q": layer.q.weight.detach().cpu().t().tolist(),
                "w_k": layer.k.weight.detach().cpu().t().tolist(),
                "w_v": layer.v.weight.detach().cpu().t().tolist(),
                "w_o": layer.o.weight.detach().cpu().t().tolist(),
                "w1": layer.ff1.weight.detach().cpu().t().tolist(),
                "w2": layer.ff2.weight.detach().cpu().t().tolist(),
                "norm1_gamma": layer.norm1.weight.detach().cpu().tolist(),
                "norm1_beta": layer.norm1.bias.detach().cpu().tolist(),
                "norm2_gamma": layer.norm2.weight.detach().cpu().tolist(),
                "norm2_beta": layer.norm2.bias.detach().cpu().tolist(),
            }
        )

    return {
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "max_seq_len": max_seq_len,
            "num_layers": num_layers,
            "ff_hidden_dim": ff_hidden_dim,
        },
        "embedding": model.embedding.weight.detach().cpu().tolist(),
        "layers": exported_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate educational Transformer weights")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--vocab",
        type=Path,
        default=repo_root / "model_tools" / "vocab.json",
        help="Path to the shared vocabulary JSON file.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=repo_root / "model_tools" / "training_corpus.txt",
        help="Path to the UTF-8 training corpus text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "app" / "src" / "main" / "assets" / "mini_transformer_weights.json",
        help="Where to write mini_transformer_weights.json.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic output.")
    parser.add_argument("--d-model", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of decoder layers.")
    parser.add_argument("--max-seq-len", type=int, default=24, help="Maximum sequence length.")
    parser.add_argument("--ff-hidden-dim", type=int, default=64, help="Feed-forward hidden dimension.")
    parser.add_argument("--train-steps", type=int, default=200, help="Toy next-token training steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="AdamW learning rate.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    vocab, stoi, itos = load_vocab(args.vocab)
    if len(stoi) != len(vocab) or len(itos) != len(vocab):
        raise ValueError("Vocabulary contains duplicate entries.")

    required_tokens = ("[PAD]", "[BOS]", "[EOS]", "[UNK]")
    missing_tokens = [token for token in required_tokens if token not in stoi]
    if missing_tokens:
        raise ValueError(f"Vocabulary is missing required tokens: {missing_tokens}")

    corpus_lines = load_training_corpus(args.corpus)

    model = MiniTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        ff_hidden_dim=args.ff_hidden_dim,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
    )

    train_model(
        model=model,
        corpus_lines=corpus_lines,
        stoi=stoi,
        steps=args.train_steps,
        lr=args.learning_rate,
    )

    exported = export_weights(
        model=model,
        vocab_size=len(vocab),
        d_model=args.d_model,
        ff_hidden_dim=args.ff_hidden_dim,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    print(f"Exported weights to {args.output}")


if __name__ == "__main__":
    main()
