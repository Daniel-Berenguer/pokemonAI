import time
import pickle
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from turnEncoder import TurnEncoder, TurnEncoderConfig


@dataclass
class RunConfig:
    # Used for sweep.py
    name: str = "run"
    model: TurnEncoderConfig = field(default_factory=TurnEncoderConfig)
    lr: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    epochs: int = 3          # can be fractional for quick sweep trials
    check_interval: int = 400
    seed: int = 0


def load_data(path="data/data.pickle"):
    with open(path, "rb") as file:
        X_train, Y_train, X_test, Y_test = pickle.load(file)
    return X_train, Y_train, X_test, Y_test


def evaluate(model, test_loader, loss_f, Y_test, device):
    model.eval()
    with torch.no_grad():
        all_logits = []
        for boardSideTest, boardFeatTest, pokeIntTest, pokeFeatTest, moveIntTest, moveFeatTest, y_batch in test_loader:
            boardSideTest = boardSideTest.to(device)
            boardFeatTest = boardFeatTest.to(device)
            pokeIntTest = pokeIntTest.to(device)
            pokeFeatTest = pokeFeatTest.to(device)
            moveIntTest = moveIntTest.to(device)
            moveFeatTest = moveFeatTest.to(device)

            logits = model.forward(pokeIntTest, pokeFeatTest, moveIntTest, moveFeatTest, boardSideTest, boardFeatTest)
            all_logits.append(logits.cpu())

        logits = torch.cat(all_logits)
        loss = loss_f(logits, Y_test)
        probs = F.sigmoid(logits)
        predicts = torch.round(probs)
        correct = torch.eq(predicts, Y_test).sum()
        acc = correct / Y_test.size(0)
    model.train()
    return loss.item(), acc.item()


def train_one_run(run_cfg: RunConfig, X_train, Y_train, X_test, Y_test, device, verbose=True):
    """Trains one model according to run_cfg and returns (model, history dict, final_metrics dict)."""
    torch.manual_seed(run_cfg.seed)

    n = Y_train.size(0)

    model = TurnEncoder(run_cfg.model).to(device)
    model.train()

    optimiser = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay)
    loss_f = nn.BCEWithLogitsLoss()

    iters = int((n * run_cfg.epochs) / run_cfg.batch_size)

    test_dataset = TensorDataset(X_test[0], X_test[1], X_test[2], X_test[3], X_test[4], X_test[5], Y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    train_dataset = TensorDataset(X_train[0], X_train[1], X_train[2], X_train[3], X_train[4], X_train[5], Y_train)
    train_loader = DataLoader(train_dataset, batch_size=run_cfg.batch_size, shuffle=True)

    history = {"step": [], "train_loss": [], "train_acc": [],
               "test_step": [], "test_loss": [], "test_acc": []}

    step_count = 0
    loss_avg = 0.0
    train_correct = 0
    train_total = 0
    start_time = time.time()

    for epoch in range(run_cfg.epochs):
        for boardSideBatch, boardFeatBatch, pokeIntBatch, pokeFeatBatch, moveIntBatch, moveFeatBatch, Y_batch in train_loader:
            boardSideBatch = boardSideBatch.to(device)
            boardFeatBatch = boardFeatBatch.to(device)
            pokeIntBatch = pokeIntBatch.to(device)
            pokeFeatBatch = pokeFeatBatch.to(device)
            moveIntBatch = moveIntBatch.to(device)
            moveFeatBatch = moveFeatBatch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model.forward(pokeIntBatch, pokeFeatBatch, moveIntBatch, moveFeatBatch, boardSideBatch, boardFeatBatch)
            loss = loss_f(logits, Y_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_avg += loss.item()
            step_count += 1

            with torch.no_grad():
                train_preds = torch.round(F.sigmoid(logits))
                train_correct += torch.eq(train_preds, Y_batch).sum().item()
                train_total += Y_batch.size(0)

            if step_count % run_cfg.check_interval == 0:
                loss_avg /= run_cfg.check_interval

                train_acc = train_correct / max(train_total, 1)

                history["step"].append(step_count)
                history["train_loss"].append(loss_avg)
                history["train_acc"].append(train_acc)

                test_loss, test_acc = evaluate(model, test_loader, loss_f, Y_test, device)
                history["test_step"].append(step_count)
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc)

                if verbose:
                    msg = f"[{run_cfg.name}] iter {step_count}/{iters}  train_loss={loss_avg:.4f}  train_acc={train_acc:.4f}"
                    if step_count % run_cfg.check_interval == 0:
                        msg += f"  test_loss={history['test_loss'][-1]:.4f}  test_acc={history['test_acc'][-1]:.4f}"
                    print(msg)

                loss_avg = 0.0
                train_correct = 0
                train_total = 0

    final_test_loss, final_test_acc = evaluate(model, test_loader, loss_f, Y_test, device)
    elapsed = time.time() - start_time

    final_metrics = {
        "name": run_cfg.name,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": elapsed,
        "config": asdict(run_cfg),
    }
    return model, history, final_metrics
