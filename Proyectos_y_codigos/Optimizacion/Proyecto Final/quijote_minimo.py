import math
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_positive_pairs(sequence, window_size=2):
    pairs = []
    for t, center in enumerate(sequence):
        left = max(0, t - window_size)
        right = min(len(sequence), t + window_size + 1)
        for s in range(left, right):
            if s != t:
                pairs.append((center, sequence[s]))
    return pairs


def tokenize_spanish(text):
    return re.findall(r"[a-záéíóúñü]+", text.lower())


def build_quijote_dataset():
    text = """
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo
    de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Del poco dormir y del mucho leer
    se le secó el cerebro, de manera que vino a perder el juicio. Llenósele la fantasía de todo aquello que leía
    en los libros de caballerías, así de encantamientos como de pendencias, batallas, desafíos, heridas, requiebros,
    amores, tormentas y disparates imposibles.

    Y cuando estuvo del todo loco, le pareció convenible hacerse caballero andante y salir por el mundo con sus armas
    y caballo a buscar aventuras. Quiso llamarse don Quijote de la Mancha, y a su rocín quiso llamar Rocinante,
    nombre a su parecer alto, sonoro y significativo. Buscó también una dama de quien enamorarse, porque el caballero
    andante sin amores era árbol sin hojas y sin fruto. A esta dama vino a llamarla Dulcinea del Toboso.

    En esto, descubrieron treinta o cuarenta molinos de viento que hay en aquel campo, y así como don Quijote los vio,
    dijo a su escudero Sancho que la ventura iba guiando sus cosas mejor de lo que acertara a desear; porque allí se
    descubrían gigantes con quien pensaba hacer batalla. Respondió Sancho que aquellos que allí se parecían no eran
    gigantes, sino molinos de viento, y lo que en ellos parecían brazos eran las aspas.

    No obstante, don Quijote arremetió a todo galope con la lanza en ristre contra el primer molino. Le acompañaba
    Sancho Panza, fiel escudero, que procuraba advertirle del engaño. Después del golpe, quedaron en el relato
    Rocinante, Sancho, Dulcinea, los molinos y la obstinación de don Quijote como símbolos inseparables de la novela.
    """

    stopwords = {
        "de", "la", "que", "y", "el", "en", "a", "los", "del", "se", "le", "lo", "por", "con", "como",
        "no", "ha", "un", "una", "su", "sus", "al", "era", "sin", "todo", "también", "allí", "aquellos",
        "aquello", "asi", "así", "hay", "muy", "o", "pero", "ya", "porque", "quien", "qué", "fue", "eran",
        "contra", "después", "las", "esto", "aquel", "aquella", "aquí", "allá", "este", "esta", "estos",
        "estas", "uno"
    }

    vocab = [
        "quijote", "sancho", "rocinante", "dulcinea", "molinos", "viento", "caballero",
        "aventuras", "mancha", "gigantes", "batalla", "escudero", "novela", "juicio"
    ]

    counts = Counter(tok for tok in tokenize_spanish(text) if tok not in stopwords and len(tok) > 2)
    vocab = [tok for tok in vocab if counts[tok] > 0]
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    sequence = [token_to_id[tok] for tok in tokenize_spanish(text) if tok in token_to_id]

    pair_counts = Counter(build_positive_pairs(sequence, window_size=2))
    positive_pairs = torch.tensor(list(pair_counts.keys()), dtype=torch.long)
    positive_weights = torch.tensor(list(pair_counts.values()), dtype=torch.float32)
    positive_weights = positive_weights / positive_weights.mean()

    return {
        "name": "quijote_clasico",
        "vocab": vocab,
        "id_to_token": {i: tok for tok, i in token_to_id.items()},
        "positive_pairs": positive_pairs,
        "positive_weights": positive_weights,
        "pair_counts": pair_counts,
        "n_tokens": len(vocab),
    }


def sample_negative_pairs(num_pairs, n_tokens):
    centers = torch.randint(low=0, high=n_tokens, size=(num_pairs,))
    contexts = torch.randint(low=0, high=n_tokens, size=(num_pairs,))
    return torch.stack([centers, contexts], dim=1)


def data_loss(E, positive_pairs, positive_weights, num_negative=256):
    i_pos = positive_pairs[:, 0]
    j_pos = positive_pairs[:, 1]
    logits_pos = (E[i_pos] * E[j_pos]).sum(dim=1)
    loss_pos = -(positive_weights * F.logsigmoid(logits_pos)).mean()

    neg_pairs = sample_negative_pairs(num_negative, E.shape[0])
    i_neg = neg_pairs[:, 0]
    j_neg = neg_pairs[:, 1]
    logits_neg = (E[i_neg] * E[j_neg]).sum(dim=1)
    loss_neg = -F.logsigmoid(-logits_neg).mean()
    return loss_pos + loss_neg


def orthogonality_penalty(E):
    gram = E.T @ E
    identity = torch.eye(E.shape[1], dtype=E.dtype, device=E.device)
    return torch.sum((gram - identity) ** 2)


def norm_violation_penalty(E, c):
    squared_norms = torch.sum(E ** 2, dim=1)
    return torch.sum(torch.clamp(squared_norms - c, min=0.0) ** 2)


def log_barrier_norm(E, c, eps=1e-8):
    squared_norms = torch.sum(E ** 2, dim=1)
    slack = c - squared_norms
    if torch.any(slack <= 0):
        return torch.tensor(float("inf"), dtype=E.dtype, device=E.device)
    return -torch.sum(torch.log(slack + eps))


def summarize_constraints(E, c):
    norms_sq = torch.sum(E ** 2, dim=1)
    gram = E.T @ E
    return {
        "orth_error": torch.linalg.norm(gram - torch.eye(E.shape[1])).item(),
        "max_norm_sq": norms_sq.max().item(),
        "min_slack": (c - norms_sq).min().item(),
    }


def train_penalty(dataset, dim=2, c=1.0, alpha=1.0, beta=10.0, lr=0.03, epochs=900):
    E = torch.nn.Parameter(0.1 * torch.randn(dataset["n_tokens"], dim))
    optimizer = torch.optim.Adam([E], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (
            data_loss(E, dataset["positive_pairs"], dataset["positive_weights"])
            + alpha * orthogonality_penalty(E)
            + beta * norm_violation_penalty(E, c)
        )
        loss.backward()
        optimizer.step()

    return E.detach()


def train_barrier(dataset, dim=2, c=1.0, alpha=1.0, mu=0.03, lr=0.01, epochs=900):
    E = torch.nn.Parameter(0.15 * torch.randn(dataset["n_tokens"], dim))
    optimizer = torch.optim.Adam([E], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (
            data_loss(E, dataset["positive_pairs"], dataset["positive_weights"])
            + alpha * orthogonality_penalty(E)
            + mu * log_barrier_norm(E, c)
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            norms = torch.linalg.norm(E, dim=1, keepdim=True)
            max_allowed = math.sqrt(c) * 0.999
            scale = torch.clamp(max_allowed / (norms + 1e-12), max=1.0)
            E.mul_(scale)

    return E.detach()


def nearest_neighbors(E, dataset, top_k=3):
    with torch.no_grad():
        normalized = E / torch.linalg.norm(E, dim=1, keepdim=True).clamp_min(1e-12)
        similarity = normalized @ normalized.T

    neighbors = {}
    for idx, token in dataset["id_to_token"].items():
        sims = similarity[idx].clone()
        sims[idx] = -1e9
        top_indices = torch.topk(sims, k=min(top_k, len(sims) - 1)).indices.tolist()
        neighbors[token] = [dataset["id_to_token"][j] for j in top_indices]
    return neighbors


def plot_embeddings(E_penalty, E_barrier, dataset, c=1.0, output_path="embeddings_quijote.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, E, title in [
        (axes[0], E_penalty, "Penalizacion cuadratica"),
        (axes[1], E_barrier, "Barrera logaritmica"),
    ]:
        E_np = E.numpy()
        ax.scatter(E_np[:, 0], E_np[:, 1], s=45)
        for idx, label in dataset["id_to_token"].items():
            ax.annotate(label, (E_np[idx, 0], E_np[idx, 1]), xytext=(5, 5), textcoords="offset points", fontsize=9)

        circle = plt.Circle((0, 0), math.sqrt(c), fill=False, linestyle="--")
        ax.add_patch(circle)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")


def main():
    dataset = build_quijote_dataset()
    E_penalty = train_penalty(dataset)
    E_barrier = train_barrier(dataset)

    penalty_summary = summarize_constraints(E_penalty, c=1.0)
    barrier_summary = summarize_constraints(E_barrier, c=1.0)

    print("Vocabulario:", dataset["vocab"])
    print()
    print("Resultados finales")
    print("-" * 72)
    print(f"{'metodo':<15}{'data_loss':>12}{'orth_error':>15}{'max_norm_sq':>15}{'min_slack':>15}")
    print(
        f"{'penalizacion':<15}"
        f"{data_loss(E_penalty, dataset['positive_pairs'], dataset['positive_weights']):>12.4f}"
        f"{penalty_summary['orth_error']:>15.4f}"
        f"{penalty_summary['max_norm_sq']:>15.4f}"
        f"{penalty_summary['min_slack']:>15.4f}"
    )
    print(
        f"{'barrera':<15}"
        f"{data_loss(E_barrier, dataset['positive_pairs'], dataset['positive_weights']):>12.4f}"
        f"{barrier_summary['orth_error']:>15.4f}"
        f"{barrier_summary['max_norm_sq']:>15.4f}"
        f"{barrier_summary['min_slack']:>15.4f}"
    )

    print()
    print("Vecinos cercanos con barrera")
    for token, neighbors in nearest_neighbors(E_barrier, dataset).items():
        print(f"{token:>12} -> {', '.join(neighbors)}")

    plot_embeddings(E_penalty, E_barrier, dataset)
    print()
    print("Grafica guardada en embeddings_quijote.png")


if __name__ == "__main__":
    main()
