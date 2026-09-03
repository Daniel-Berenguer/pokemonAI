import json
import random

import torch
import matplotlib.pyplot as plt

from turnEncoder import TurnEncoderConfig
from train_utils import RunConfig, load_data, train_one_run


def build_configs(n_trials=10, seed=2):
    random.seed(seed)

    configs = []

    for i in range(n_trials):
        move_dim = 32
        move_hidden = 128
        poke_emb = 64
        ab_emb = 16
        it_emb = 16
        board_dim = 16
        poke_dim = random.choice([96, 128, 192, 256])
        n_heads = random.choice([8,16,32])
        hidden_lay_1 = random.randint(64, 128)
        hidden_lay_2 = 32
        poke_dropout = 0.25
        self_att_dropout = 0.45
        cross_att_dropout = 0.45
        mlp_dropout1 = 0.35
        mlp_dropout2 = 0.1
        lr=2.5e-5
        weight_decay=1.5e-4
        batch_size = 64
        

        # Ensure divisibility
        if poke_dim % n_heads != 0:
            continue


        configs.append(RunConfig(
                name=f"trial_{i:02d}",
                model=TurnEncoderConfig(
                    MOVE_DIM=move_dim,
                    MOVE_HIDDEN=move_hidden,
                    POKE_EMB=poke_emb,
                    AB_EMB=ab_emb,
                    IT_EMB=it_emb,
                    BOARD_DIM=board_dim,
                    POKE_DIM=poke_dim,
                    N_HEADS=n_heads,
                    HIDDEN_LAY_1=hidden_lay_1,
                    HIDDEN_LAY_2=hidden_lay_2,
                    poke_dropout=poke_dropout,
                    self_att_dropout=self_att_dropout,
                    cross_att_dropout=cross_att_dropout,
                    mlp_dropout1=mlp_dropout1,
                    mlp_dropout2=mlp_dropout2,
                ),

                lr=lr,      # 1e-5 to 1e-3 (log-uniform)
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=2
            )
        )

    return configs



def run_sweep(configs, data_path="data/data.pickle", out_dir="data/sweep"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train, Y_train, X_test, Y_test = load_data(data_path)

    all_histories = {}
    all_metrics = []

    for cfg in configs:
        print(f"\n=== Training config: {cfg.name} ===")
        print(cfg)
        try:
            _, history, metrics = train_one_run(cfg, X_train, Y_train, X_test, Y_test, device)
            all_histories[cfg.name] = history
            all_metrics.append(metrics)
        except Exception as e:
            print(f"FAILED config {cfg.name}: {e}")
            all_metrics.append({"name": cfg.name, "error": str(e), "config": cfg.__dict__})

    # Persist raw results so you can re-plot/re-analyze without re-training
    with open(f"{out_dir}/histories.pickle", "wb") as f:
        import pickle
        pickle.dump(all_histories, f)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Sorted leaderboard by final test accuracy
    ranked = sorted(
        [m for m in all_metrics if "final_test_acc" in m],
        key=lambda m: m["final_test_acc"], reverse=True,
    )
    print("\n=== Leaderboard (by final test accuracy) ===")
    for m in ranked:
        print(f"{m['name']:20s}  test_acc={m['final_test_acc']:.4f}  test_loss={m['final_test_loss']:.4f}"
              f"  params={m['n_params']:,}  time={m['elapsed_sec']:.0f}s")

    plot_comparison(all_histories, out_dir)
    return all_histories, all_metrics


def plot_comparison(all_histories, out_dir="data/sweep"):
    """Overlay every run's train/test loss and accuracy so they're directly comparable."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_train_loss, ax_test_loss), (ax_train_acc, ax_test_acc) = axes

    for name, history in all_histories.items():
        ax_train_loss.plot(history["step"], history["train_loss"], label=name)
        ax_test_loss.plot(history["test_step"], history["test_loss"], label=name)
        ax_train_acc.plot(history["step"], history["train_acc"], label=name)
        ax_test_acc.plot(history["test_step"], history["test_acc"], label=name)

    for ax, title in [
        (ax_train_loss, "Train loss"), (ax_test_loss, "Test loss"),
        (ax_train_acc, "Train accuracy"), (ax_test_acc, "Test accuracy"),
    ]:
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/sweep_comparison.png", dpi=150)
    print(f"\nSaved comparison plot to {out_dir}/sweep_comparison.png")


if __name__ == "__main__":
    configs = build_configs()
    all_histories, all_metrics = run_sweep(configs)
    print(all_metrics)