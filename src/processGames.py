from Board import Board
from boardToTensors import board2tensor
import torch
import pickle
import random

def processGame(text, tensors, labels, augment=True):
    

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

        if line.startswith("|drag|"):
            board.switch(line[len("|drag|"):])

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

        if line.startswith("|-status|"):
            board.startStatus(line[len("|-status|"):])

        if line.startswith("|-curestatus|"):
            board.endStatus(line[len("|-curestatus|"):])

        if line.startswith("|-singleturn|") and "Protect" in line:
            board.protected(line[len("|-singleturn|"):])

        if line.startswith("|-start|") and "Substitute" in line:
            board.startSub(line[len("|-start|"):])

        if line.startswith("|-end|") and "Substitute" in line:
            board.endSub(line[len("|-end|"):])

        if line.startswith("|-start|") and "perish" in line:
            board.perish(line[len("|-start|"):])

        if line.startswith("|-detailschange|"):
            board.mega(line[len("|-detailschange|"):])


        if line.startswith("|turn|"):
            if not firstTurn:
                board.nextTurn()
            else:
                firstTurn = False
            
            # Converts board to tensor and adds to data
            t = board2tensor(board)
            for i, tensor in enumerate(t):
                tensors[i].append(tensor)
            try:
                labels.append(torch.tensor(board.winner))
            except:
                for i, _ in enumerate(t):
                    tensors[i].pop()
            if augment:
                # Switches sides of board (data augmentation)
                board.switchSides()
                # Tensorizes and stores
                h = board2tensor(board)
                for i, tensor in enumerate(h):
                    tensors[i].append(tensor)
                try:
                    labels.append(torch.tensor(board.winner))
                except:
                    for i, _ in enumerate(t):
                        tensors[i].pop()
                # Switches back
                board.switchSides()
        

if __name__ == "__main__":
    import os
    total = 0
    errors = 0
    train_tensors = [[],[],[],[],[],[]]
    train_labels = []
    test_tensors = [[],[],[],[],[],[]]
    test_labels = []
    filenames = os.listdir("data/games")
    random.shuffle(filenames)

    N_FILES = len(filenames)
    N_TEST = N_FILES*0.15

    error_dict = dict()
    for i,filename in enumerate(filenames):
        with open(f"data/games/{filename}", encoding="utf-8") as file:
            text = file.read()
        print(filename)
        total += 1
        try:
            if i <= N_TEST:
                processGame(text, test_tensors, test_labels)
            else:
                processGame(text, train_tensors, train_labels)
        except Exception as e:
            print(e)
            err = str(e)
            if err not in error_dict:
                error_dict[err] = 1
            else:
                error_dict[err] += 1
            errors += 1
    processed = total-errors

    print(f"Processed {processed}/{total}")
    print(f"{processed/total:.2%}")
    print(f"Data points: {len(train_tensors[0]) + len(test_tensors[0])}")


    X_train = [torch.stack(tensor) for tensor in train_tensors]
    Y_train = torch.stack(train_labels).float()

    X_test = [torch.stack(tensor) for tensor in test_tensors]
    Y_test = torch.stack(test_labels).float()

    print(error_dict)

    with open("data/data.pickle", "wb") as file:
        pickle.dump([X_train, Y_train, X_test, Y_test], file)
