from Board import Board
from Pokemon import Pokemon
import torch

ab2ix = dict()

with open("data/abilities.csv", "r") as file:
    for i, ab in enumerate(file.read().split("\n")):
        ab2ix[ab] = i

items2ix = dict()

with open("data/items.csv", "r") as file:
    for i, item in enumerate(file.read().split("\n")):
        items2ix[item] = i


moves2ix = dict()

with open("data/moves.csv", "r") as file:
    for i, moveStats in enumerate(file.read().split("\n")):
        move = moveStats.split(",")[0]
        moves2ix[move] = i


pokemon2ix = dict()

with open("data/pokemon-stats.csv", "r") as file:
    for i, pokeStats in enumerate(file.read().split("\n")):
        poke = pokeStats.split(",")[0]
        pokemon2ix[poke] = i

type2ix = dict()

with open("data/types.csv", "r") as file:
    for i, type in enumerate(file.read().split("\n")):
        type2ix[type] = i

weather2ix = {"none" : 0,
              "RainDance" : 1,
              "SunnyDay" : 2,
              "Sandstorm": 3,
              "Snowscape" : 4}

terrain2ix = {"none" : 0,
           "Grassy Terrain" : 1,
           "Psychic Terrain" : 2,
           "Electric Terrain" : 3,
           "Misty Terrain" : 4}

class2ix = {"Status" : 0,
            "Physical" : 1,
            "Special" : 2}

status2ix = {"none" : 0,
             "brn" : 1,
             "par" : 2,
             "slp" : 3,
             "frz" : 4,
             "psn" : 5,
             "tox" : 6}


def append(array, x):
    if isinstance(x, list):
        for y in x:
            if isinstance(y, str):
                y = int(y)
            append(array, y)
    else:
        array.append(x)


def onehot(size, ix):
    arr = [0] * size
    arr[ix] = 1
    return arr


def board2tensor(board: Board):
    array = []
    arrayInt = []

    append(arrayInt, board.tailwinds) # 2 ints (0-5)
    append(arrayInt, board.trickroom) # 1 int (0-5)
    append(arrayInt, weather2ix[board.weather[0]]) # 1 int (0-4)
    append(arrayInt, terrain2ix[board.terrain[0]]) # 1 int (0-4)
    
    append(array, board.gravity/5)
    append(array, board.weather[1]/8)
    append(array, board.terrain[1]/8)
    append(array, [x[1]/8 for x in board.auroraveils])
    append(array, [x[0] for x in board.auroraveils])
    append(array, [x[1]/8 for x in board.reflects])
    append(array, [x[0] for x in board.reflects])
    append(array, [x[1]/8 for x in board.lightscreens])
    append(array, [x[0] for x in board.lightscreens])


    

    moveFeats = []
    moveInts = []
    pokeInts = []
    pokeFeats = []
    for i,team in enumerate(board.pokemon):
        moveFeats.append([])
        moveInts.append([])
        pokeFeats.append([])
        pokeInts.append([])
        for poke in team:
            pokeInt, pokeFeat, moveInt, moveFeat = pokemon2array(poke)
            moveFeats[i].append(moveFeat)
            moveInts[i].append(moveInt)
            pokeInts[i].append(pokeInt)
            pokeFeats[i].append(pokeFeat)

    boardIntTensor = torch.tensor(arrayInt, dtype=torch.long) # Shape (5) [tw1, tw2, tr, weather, terrain]
    boardTensor = torch.tensor(array, dtype=torch.float) # Shape (boardFDim=15)
    pokeIntsTensor = torch.tensor(pokeInts, dtype=torch.long) # Shape (2, 6, 6) [poke, item, ab, typ1, typ2, tera]
    pokeFeatsTensor = torch.tensor(pokeFeats, dtype=torch.float) # Shape (2, 6, pokeFeatDim=24)
    moveIntsTensor = torch.tensor(moveInts, dtype=torch.long) # Move Types (2, 6, 4, 2) [moveName, moveType]
    moveFeatsTensor = torch.tensor(moveFeats, dtype=torch.float) # Move Features (2, 6, 4, moveFeatDim)

    return (boardIntTensor, boardTensor, pokeIntsTensor, pokeFeatsTensor, moveIntsTensor, moveFeatsTensor)


def pokemon2array(poke: Pokemon):
    intFeats = []
    feats = []

    intFeats.append(pokemon2ix[poke.name])
    intFeats.append(items2ix[poke.item])
    intFeats.append(ab2ix[poke.ability])
    intFeats += [type2ix[poke.stats[0]], type2ix[poke.stats[1]], type2ix[poke.teratype]]
    stats = poke.stats[2:]
    # [HP, ATK, DEF, SPA, SPD, SPE]
    dividers = [255, 190, 230, 195, 230, 200]
    for i, div in enumerate(dividers):
        stats[i] = int(stats[i]) / div

    append(feats, stats)
    boosts = poke.boosts
    for i, _ in enumerate(boosts):
        boosts[i] /= 6
    append(feats, boosts)
    feats.append(poke.hp/100)
    feats.append(poke.tera)
    feats.append(poke.justProtected)
    feats.append(poke.fnt)
    feats.append(poke.sub)
    feats.append(poke.lostItem)
    feats.append(poke.team)
    append(feats, onehot(len(status2ix), status2ix[poke.status]))

    moveFeats = []
    moveInts = []

    for move in poke.moves:
        mvFt = []
        mvInt = []
        mvInt.append(moves2ix[move[0]])
        mvInt.append(type2ix[move[1]])
        append(mvFt, onehot(len(class2ix), class2ix[move[2]]))
        mvFt.append(int(move[3])/150)
        mvFt.append(int(move[4])/100)
        mvFt.append(int(move[5])/8)
        moveFeats.append(mvFt)
        moveInts.append(mvInt)
    
    return intFeats, feats, moveInts, moveFeats


if __name__ == "__main__":
    with open("data/games/gen9vgc2025reghbo3-2414586101") as file:
        lines = file.read().split("\n")

    board = Board()


    for line in lines:
            if line.startswith("|showteam|"):
                board.loadPokemon(line)

    board2tensor(board)