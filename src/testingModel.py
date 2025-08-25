from processGames import processGame
import torch
from model import Model

with open("data/testGame", "r", encoding="utf-8") as file:
    text = file.read()

tensors = ([], [], [], [], [], [])
labels = []

processGame(text, tensors, labels, augment=False)

X = [torch.stack(tensor) for tensor in tensors]
Y = torch.stack(labels).float()

boardInt = X[0]
boardFeat = X[1]
pokeInt = X[2]
pokeFeat = X[3]
moveInt = X[4]
moveFeat = X[5]

model = Model()
with open("data/model_state_dict", "rb") as file:
    state_dict = torch.load(file)
model.load_state_dict(state_dict)

logits = model.forward(boardInt, boardFeat, pokeInt, pokeFeat, moveInt, moveFeat)
probs = torch.sigmoid(logits)

#print(Y)
print(probs)

