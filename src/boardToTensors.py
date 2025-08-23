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
    sides = []
    tailwinds = [onehot(6, board.tailwinds[0]), onehot(6, board.tailwinds[1])]

    trickroom = onehot(6, board.trickroom)

    append(array, tailwinds)
    append(array, trickroom)
    sides[0].append()
    append(array, board.gravity)
    append(array, onehot(len(weather2ix), weather2ix[board.weather[0]]))
    append(array, board.weather[1])
    append(array, onehot(len(terrain2ix), terrain2ix[board.terrain[0]]))
    append(array, board.terrain[1])
    append(array, board.trickroom)
    append(array, board.gravity)
    append(array, board.auroraveils)
    append(array, board.reflects)
    append(array, board.lightscreens)
    append(array, board.shown)
    for team in board.pokemon:
        for poke in team:
            append(array, pokemon2array(poke))
    tensor = torch.tensor(array)
    return tensor


def pokemon2array(poke: Pokemon):
    array = []
    append(array, onehot(len(pokemon2ix), pokemon2ix[poke.name]))
    append(array, onehot(len(items2ix), items2ix[poke.item]))
    append(array, onehot(len(ab2ix), ab2ix[poke.ability]))
    for move in poke.moves:
        append(array, onehot(len(moves2ix), moves2ix[move[0]]))
        append(array, onehot(len(type2ix), type2ix[move[1]]))
        append(array, onehot(len(class2ix), class2ix[move[2]]))
        append(array, move[3:])
    append(array, onehot(len(type2ix), type2ix[poke.stats[0]]))
    append(array, onehot(len(type2ix), type2ix[poke.stats[1]]))
    append(array, poke.stats[2:])
    append(array, poke.boosts)
    append(array, poke.hp)
    append(array, onehot(len(type2ix), type2ix[poke.teratype]))
    append(array, poke.tera)
    append(array, poke.justProtected)
    append(array, poke.fnt)
    append(array, poke.sub)
    append(array, onehot(4, poke.perish))
    append(array, poke.lostItem)
    return array


if __name__ == "__main__":
    with open("data/games/gen9vgc2025reghbo3-2414586101") as file:
        lines = file.read().split("\n")

    board = Board()


    for line in lines:
            if line.startswith("|showteam|"):
                board.loadPokemon(line)

    board2tensor(board)