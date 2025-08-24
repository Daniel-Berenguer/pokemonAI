import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

Nin = 27889

cuda = torch.device('cuda')

class LinearRELU(nn.Module):
    def __init__(self, nin, nout) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.ones(nin, nout))
        self.b = nn.Parameter(torch.zeros(nout))
        self.bn = nn.BatchNorm1d(nout)
        nn.init.kaiming_normal_(self.W, mode="fan_in", nonlinearity="relu")
        

    def forward(self, x):
        x = x @ self.W + self.b
        x = F.relu(self.bn.forward(x))
        return x


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = LinearRELU(680, 1024)
        self.lin2 = LinearRELU(1024, 2048)
        self.lin3 = LinearRELU(2048, 1024)
        self.lin4 = LinearRELU(1024, 256)
        self.lin5 = LinearRELU(256, 64)
        self.linear = [self.lin1, self.lin2, self.lin3, self.lin4, self.lin5]
        self.W = nn.Parameter(torch.randn(64))
        self.b = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        for linear in self.linear:
            x = linear.forward(x)
        return x @ self.W + self.b
    
model = Model().to(cuda)
nEl = 0
for p in model.parameters():
    nEl += p.numel()

print(nEl)

with open("data/data.pickle", "rb") as file:
    X, Y = pickle.load(file)
    X = torch.cat([T.flatten(start_dim=1).float() for T in X], dim=-1) 



BATCH_SIZE = 16

optimizer = torch.optim.Adam(model.parameters())


# Shuffle
indices = torch.randperm(X.size(0))  # random permutation of indices
X_shuffled = X[indices]
Y_shuffled = Y[indices]

n = int(Y_shuffled.size(0)*0.85)

print(X_shuffled.shape)
print(Y_shuffled.shape)
print(n)

X_train = X_shuffled[:n]
Y_train = Y_shuffled[:n]
X_test = X_shuffled[n:]
Y_test = Y_shuffled[n:]

print(Y_train.shape)
print(Y_test.shape)

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
    print(probs)
    print(f"Test accuracy: {acc.item():.4f}")