import torch
import matplotlib.pyplot as plt

from turnEncoder import TurnEncoderConfig
from train_utils import RunConfig, load_data, train_one_run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

X_train, Y_train, X_test, Y_test = load_data("data/data.pickle")

print(f"Train Dataset size: {Y_train.size(0)}")
print(X_train[0].shape)
print(Y_train.shape)
print(X_test[0].shape)
print(Y_test.shape)


# --- This is where you plug in the winning config from your sweep ---
run_cfg = RunConfig(
    name="final",
    model=TurnEncoderConfig(),   # defaults == original hardcoded values; edit here after sweeping
    lr=1e-4,
    weight_decay=0.05,
    batch_size=64,
    epochs=2,
    check_interval=200,
)

model, history, final_metrics = train_one_run(run_cfg, X_train, Y_train, X_test, Y_test, device)

print(f"Number of params: {final_metrics['n_params']}")
print(f"Test loss: {final_metrics['final_test_loss']:.4f}")
print(f"Test accuracy: {final_metrics['final_test_acc']:.4f}")

with open("data/model_state_dict", "wb") as file:
    torch.save(model.state_dict(), file)

import pickle
with open("data/history.pickle", "wb") as file:
    pickle.dump(history, file)

# Plot train/test loss and accuracy
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

ax_loss.plot(history["step"], history["train_loss"], label="Train loss")
ax_loss.plot(history["test_step"], history["test_loss"], label="Test loss")
ax_loss.set_xlabel("Step")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Loss")
ax_loss.legend()
ax_loss.grid(alpha=0.3)

ax_acc.plot(history["step"], history["train_acc"], label="Train accuracy")
ax_acc.plot(history["test_step"], history["test_acc"], label="Test accuracy")
ax_acc.set_xlabel("Step")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy")
ax_acc.legend()
ax_acc.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("data/training_curves.png", dpi=150)
print("Saved training curves to data/training_curves.png")
plt.show()
