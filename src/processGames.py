from Board import Board
from boardToTensors import board2tensor
import torch
import pickle

def processGame(filename, tensors, labels):
    with open(f"data/games/{filename}", encoding="utf-8") as file:
        text = file.read()

    lines = text.split("\n")

    # i -> Line index

    board = Board()

    firstTurn = True

    # First we get the winner
    p1 = None
    p2 = None
    for line in lines:
        if line.startswith("|player|p1|"):
            p1 = line.split("|")[3]
        elif line.startswith("|player|p2|"):
            p2 = line.split("|")[3]
        elif line.startswith("|win|"):
            winnerName = line.split("|")[2]
            if winnerName == p1:
                board.winner = 0
            else:
                board.winner = 1


    for line in lines:
        if line.startswith("|showteam|"):
            board.loadPokemon(line)

        if line.startswith("|switch|"):
            board.switch(line[len("|switch|"):])

        if line.startswith("|-fieldstart|"):
            board.startField(line[len("|-fieldstart|"):])

        if line.startswith("|-sidestart|"):
            board.updateSide(line[len("|-sidestart|"):], True)

        if line.startswith("|-sideend|"):
            board.updateSide(line[len("|-sideend|"):], False)

        if line.startswith("|-weather|"):
            board.startWeather(line[len("|-weather|"):])

        if line.startswith("|-boost|"):
            board.boost(line[len("|-boost|"):])

        if line.startswith("|-unboost|"):
            board.boost(line[len("|-unboost|"):])

        if line.startswith("|-damage|"):
            board.updateHP(line[len("|-damage|"):])

        if line.startswith("|-heal|"):
            board.updateHP(line[len("|-heal|"):])

        if line.startswith("|-enditem|"):
            board.endItem(line[len("|-enditem|"):])

        if line.startswith("|-fieldend|"):
            board.endItem(line[len("|-fieldend|"):])

        if line.startswith("|-singleturn|") and "Protect" in line:
            board.protected(line[len("|-singleturn|"):])

        if line.startswith("-terastallize"):
            board.tera(line[len("-terastallize"):])

        if line.startswith("|-start|") and "Substitute" in line:
            board.startSub(line[len("|-start|"):])

        if line.startswith("|-end|") and "Substitute" in line:
            board.endSub(line[len("|-end|"):])

        if line.startswith("|-start|") and "perish" in line:
            board.perish(line[len("|-start|"):])


        if line.startswith("|turn|"):
            if not firstTurn:
                board.nextTurn()
            else:
                firstTurn = False
            t = board2tensor(board)
            for i, tensor in enumerate(t):
                tensors[i].append(tensor)
                labels.append(torch.tensor(board.winner))
        

import os
total = 0
errors = 0
tensors = [[],[],[],[],[],[]]
labels = []
for filename in os.listdir("data/games"):
    print(filename)
    total += 1
    if filename != "gen9vgc2025reghbo3-2416755957":
        try:
            processGame(filename, tensors, labels)
        except Exception as e:
            print(e)
            errors += 1

processed = total-errors

print(f"Processed {processed}/{total}")
print(f"{processed/total:.2%}")
print(f"Data points: {len(tensors[0])}")

X = [torch.stack(tensor) for tensor in tensors]
Y = torch.stack(labels)

with open("data/data.pickle", "wb") as file:
    pickle.dump([X, Y], file)
