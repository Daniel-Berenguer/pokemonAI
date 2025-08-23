import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

cuda = torch.device('cuda')
    
#boardIntTensor (5) [tw1, tw2, tr, weather, terrain]
#boardTensor (boardFDim=15)
#pokeIntsTensor (2, 6, 6) [poke, item, ab, typ1, typ2, tera]
#pokeFeatsTensor (2, 6, pokeFeatDim=17)
#moveIntsTensor (2, 6, 4, 2) [moveName, moveType]
#moveFeatsTensor (2, 6, 4, moveFeatDim)

class typeEncoder(nn.Module):
    NUM_TYPES = 20
    def __init__(self, nhidden=10):
        super().__init__()
        self.emb = nn.Embedding(self.NUM_TYPES, nhidden)

    def forward(self, x):
        return self.emb(x)
    

class moveEncoder(nn.Module):
    NUM_MOVES = 382
    MOVE_FEATS_SIZE = 6
    def __init__(self, nhidden=128, typeEmbeddingSize=10):
        super().__init__()
        self.emb = nn.Embedding(self.NUM_MOVES, nhidden) # Embeds move (move name, to capture effects I have not included in features)
        self.U = nn.Parameter(torch.randn(typeEmbeddingSize, nhidden)) # Multiplies type embedding before adding to move embedding
        self.W = nn.Parameter(torch.randn(self.MOVE_FEATS_SIZE, nhidden)) # Multiplies move features (e.g power) before adding to move embedding

    def forward(self, moveInts, moveFeats, typeEncoder : typeEncoder):
        move_name = moveInts[:, :, :, 0]
        move_type = moveInts[:, :, :, 1]
        return self.emb(move_name) + typeEncoder.forward(move_type) @ self.U + moveFeats @ self.W
    
#boardIntTensor (5) [tw1, tw2, tr, weather, terrain]
#boardTensor (boardFDim=15)
#pokeIntsTensor (2, 6, 6) [poke, item, ab, typ1, typ2, tera]
#pokeFeatsTensor (2, 6, pokeFeatDim=17)
#moveIntsTensor (2, 6, 4, 2) [moveName, moveType]
#moveFeatsTensor (2, 6, 4, moveFeatDim)

class PokeEncoder(nn.Module):
    N_POKES = 256
    N_ABS = 212
    N_ITEMS = 133
    def __init__(self, nhidden=128, typeEmb=10):
        super().__init__()
        self.pokeEmb = nn.Embedding(self.N_POKES, nhidden)
        self.abEmb = nn.Embedding(self.N_ABS, nhidden)
        self.itemEmb = nn.Embedding(self.N_ITEMS, nhidden)



with open("data/data.pickle", "rb") as file:
    X, Y = pickle.load(file)
    X = X.float()
    Y = Y.float()

model = Model().to(cuda)

BATCH_SIZE = 16

optimizer = torch.optim.Adam(model.parameters())


# Shuffle
indices = torch.randperm(X.size(0))  # random permutation of indices
X_shuffled = X[indices]
Y_shuffled = Y[indices]

n = int(Y.size(0)*0.85)

X_train = X_shuffled[:n]
Y_train = Y_shuffled[:n]
X_test = X_shuffled[n:]
Y_test = Y_shuffled[n:]


EPOCHS = 15
ITERS = int((n * EPOCHS)/BATCH_SIZE)

loss_avg = 0

for i in range(1201):
    ix = torch.randint(0, X_train.size(0), (BATCH_SIZE,))  # random indices
    X_batch = X_train[ix].to(cuda)
    Y_batch = Y_train[ix].to(cuda)

    logits = model.forward(X_batch)

    loss = F.binary_cross_entropy_with_logits(logits, Y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_avg += loss.item()
    if i % 10 == 0:
        loss_avg /= 10
        print(f'Iteration [{i}/{ITERS}], Loss: {loss_avg:.4f}')
        loss_avg = 0


# Test loss
with torch.no_grad():
    X_test = X_test.to(cuda)
    Y_test = Y_test.to(cuda)
    logits = model.forward(X_test)
    loss = F.binary_cross_entropy_with_logits(logits, Y_test)
    print(f"Test loss: {loss.item():.4f}")
    probs = F.sigmoid(logits)
    predicts = torch.round(probs)
    correct = torch.eq(predicts, Y_test).sum()
    acc = correct / Y_test.size(0)
    print(f"Test accuracy: {acc.item():.4f}")