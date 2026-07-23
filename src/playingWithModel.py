import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from turnEncoder import TurnEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from processGames import processGame

cuda = torch.device('cuda')

model = TurnEncoder()

with open("data/model_state_dict", "rb") as file:
    state_dict = torch.load(file)

model.load_state_dict(state_dict)
model.to(cuda)

filenames = ["gen9championsvgc2026regmbbo3-2653589326",
"gen9championsvgc2026regmbbo3-2653588856",
"gen9championsvgc2026regmbbo3-2653589612",
"gen9championsvgc2026regmbbo3-2653588249",
"gen9championsvgc2026regmbbo3-2653587893",
"gen9championsvgc2026regmbbo3-2653587977",
"gen9championsvgc2026regmbbo3-2653586174",
"gen9championsvgc2026regmbbo3-2653586117",
"gen9championsvgc2026regmbbo3-2653586228",
"gen9championsvgc2026regmbbo3-2653584617",
"gen9championsvgc2026regmbbo3-2653585243",
"gen9championsvgc2026regmbbo3-2653583888",
"gen9championsvgc2026regmbbo3-2653582292",
"gen9championsvgc2026regmbbo3-2653584497",
"gen9championsvgc2026regmbbo3-2653584180",
"gen9championsvgc2026regmbbo3-2653583464",
"gen9championsvgc2026regmbbo3-2653581428",
"gen9championsvgc2026regmbbo3-2653579983",
"gen9championsvgc2026regmbbo3-2653580328",
"gen9championsvgc2026regmbbo3-2653578604",
"gen9championsvgc2026regmbbo3-2653576980",
"gen9championsvgc2026regmbbo3-2653575605",
"gen9championsvgc2026regmbbo3-2653574253",
"gen9championsvgc2026regmbbo3-2653574574",
"gen9championsvgc2026regmbbo3-2653572927",
"gen9championsvgc2026regmbbo3-2653572760",
"gen9championsvgc2026regmbbo3-2653571883",
"gen9championsvgc2026regmbbo3-2653570812",
"gen9championsvgc2026regmbbo3-2653570676",
"gen9championsvgc2026regmbbo3-2653569657",
"gen9championsvgc2026regmbbo3-2653567265",
"gen9championsvgc2026regmbbo3-2653568294",
"gen9championsvgc2026regmbbo3-2653568004",
"gen9championsvgc2026regmbbo3-2653567570",
"gen9championsvgc2026regmbbo3-2653566828",
"gen9championsvgc2026regmbbo3-2653564111"]

for filename in filenames:
    print(f"-----------------------------------------GAME {filename}-------------------------")
    tensors = [[],[],[],[],[],[]]
    labels = []

    with open(f"data/games/{filename}", encoding="utf-8") as file:
        text = file.read()
        processGame(text, tensors, labels, augment=False)

    X = [torch.stack(tensor) for tensor in tensors]
    Y = torch.stack(labels).float()

    boardInt = X[0].to(cuda)
    boardFeat = X[1].to(cuda)
    pokeInt = X[2].to(cuda)
    pokeFeat = X[3].to(cuda)
    moveInt = X[4].to(cuda)
    moveFeat = X[5].to(cuda)

    print(moveInt.shape)

    logits = model.forward(pokeInt, pokeFeat, moveInt, moveFeat,boardInt, boardFeat)

    output = torch.sigmoid(logits)

    print(output)
    print(Y)