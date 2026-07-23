import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from turnEncoder import TurnEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


cuda = torch.device('cuda')

# Load Data from disk
with open("data/data.pickle", "rb") as file:
        # X is boardIntTensor, boardTensor, pokeIntsTensor, pokeFeatsTensor, moveFeatsTensor, moveIntsTensor
        X_train, Y_train, X_test, Y_test = pickle.load(file)

n = Y_train.size(0)
print(f"Train Dataset size: {Y_train.size(0)}")

print(X_train[0].shape)
print(Y_train.shape)
print(X_test[0].shape)
print(Y_test.shape)


# Shuffle
indices = torch.randperm(X_train[0].size(0))  # random permutation of indices
for i, tens in enumerate(X_train):
    X_train[i] = tens[indices]
Y_train = Y_train[indices]


# Initialise Model
model = TurnEncoder().to(cuda)
model.train()
nEl = 0
for p in model.parameters():
    nEl += p.numel()
print(f"Number of params: {nEl}")


# Define optimiser and loss
optimiser = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
loss_f = nn.BCEWithLogitsLoss()

EPOCHS = 4
BATCH_SIZE = 64
ITERS = int((n * EPOCHS)/BATCH_SIZE)
CHECK_INTERVAL = 200
ITERS = 5000

test_dataset = TensorDataset(X_test[0], X_test[1], X_test[2], X_test[3], X_test[4], X_test[5], Y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

def evaluate():
    """Runs a full pass over the test set and returns (loss, accuracy)."""
    model.eval()
    with torch.no_grad():
        all_logits = []
        for boardIntTest, boardFeatTest, pokeIntTest, pokeFeatTest, moveIntTest, moveFeatTest, y_batch in test_loader:
            boardIntTest = boardIntTest.to(cuda)
            boardFeatTest = boardFeatTest.to(cuda)
            pokeIntTest = pokeIntTest.to(cuda)
            pokeFeatTest = pokeFeatTest.to(cuda)
            moveIntTest = moveIntTest.to(cuda)
            moveFeatTest = moveFeatTest.to(cuda)

            logits = model.forward(pokeIntTest, pokeFeatTest, moveIntTest, moveFeatTest, boardIntTest, boardFeatTest)
            all_logits.append(logits.cpu())  # move off GPU immediately

        logits = torch.cat(all_logits)
        loss = loss_f(logits, Y_test)
        probs = F.sigmoid(logits)
        predicts = torch.round(probs)
        correct = torch.eq(predicts, Y_test).sum()
        acc = correct / Y_test.size(0)
    model.train()
    return loss.item(), acc.item()


# Metric history for plotting
history = {
    "iter": [],
    "train_loss": [],
    "train_acc": [],
    "test_iter": [],
    "test_loss": [],
    "test_acc": [],
}

loss_avg = 0
train_correct = 0
train_total = 0

for i in range(ITERS+1):
    ix = torch.randint(0, Y_train.size(0), (BATCH_SIZE,))  # random indices
    #boardIntTensor, boardTensor, pokeIntsTensor, pokeFeatsTensor, moveFeatsTensor, moveIntsTensor
    boardIntBatch = X_train[0][ix].to(cuda)
    boardFeatBatch = X_train[1][ix].to(cuda)
    pokeIntBatch = X_train[2][ix].to(cuda)
    pokeFeatBatch = X_train[3][ix].to(cuda)
    moveIntBatch = X_train[4][ix].to(cuda)
    moveFeatBatch = X_train[5][ix].to(cuda)

    Y_batch = Y_train[ix].to(cuda)

    logits = model.forward(pokeIntBatch, pokeFeatBatch, moveIntBatch, moveFeatBatch,boardIntBatch, boardFeatBatch)

    loss = loss_f(logits, Y_batch)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    loss_avg += loss.item()

    with torch.no_grad():
            train_preds = torch.round(F.sigmoid(logits))
            train_correct += torch.eq(train_preds, Y_batch).sum().item()
            train_total += Y_batch.size(0)

    if i % CHECK_INTERVAL == 0 or i == ITERS:
        if i != 0:
            if i % CHECK_INTERVAL == 0:
                loss_avg /= CHECK_INTERVAL
            else:
                loss_avg /= (i % CHECK_INTERVAL)
        train_acc = train_correct / max(train_total, 1)
        print(f'Iteration [{i}/{ITERS}], Loss: {loss_avg:.4f}')
        print(f'Iteration [{i}/{ITERS}], Loss: {loss_avg:.4f}, Train Acc: {train_acc:.4f}')

        history["iter"].append(i)
        history["train_loss"].append(loss_avg)
        history["train_acc"].append(train_acc)
        
        loss_avg = 0
        train_correct = 0
        train_total = 0
        
    if i % CHECK_INTERVAL == 0:
            test_loss, test_acc = evaluate()
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
    
            history["test_iter"].append(i)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

# Final evaluation
final_test_loss, final_test_acc = evaluate()
print(f"Test loss: {final_test_loss:.4f}")
print(f"Test accuracy: {final_test_acc:.4f}")

with open("data/model_state_dict", "wb") as file:
    torch.save(model.state_dict(), file)

# Save metric history alongside the model, so we can re-plot later without retraining
with open("data/history.pickle", "wb") as file:
    pickle.dump(history, file)


# Plot train/test loss and accuracy
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

ax_loss.plot(history["iter"], history["train_loss"], label="Train loss")
ax_loss.plot(history["test_iter"], history["test_loss"], label="Test loss")
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Loss")
ax_loss.legend()
ax_loss.grid(alpha=0.3)

ax_acc.plot(history["iter"], history["train_acc"], label="Train accuracy")
ax_acc.plot(history["test_iter"], history["test_acc"], label="Test accuracy")
ax_acc.set_xlabel("Iteration")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy")
ax_acc.legend()
ax_acc.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("data/training_curves.png", dpi=150)
print("Saved training curves to data/training_curves.png")
plt.show()