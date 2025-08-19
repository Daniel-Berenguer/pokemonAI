import pickle
from Pokemon import Pokemon

with open("data/moves.csv", "r") as file:
        lines = file.read().split("\n")
        move_dict = dict()
        for line in lines:
             line = line.split(",")
             move_dict[line[0]] = line[1:]

with open("data/pokemon-stats.csv", "r") as file:
        lines = file.read().split("\n")
        pokemon_stats = dict()
        for line in lines:
             line = line.split(",")
             pokemon_stats[line[0]] = line[1:]

class Board:
    def __init__(self):
        self.tailwinds = [0, 0]
        self.weather = "None"
        self.terrain = "None"
        self.gravity = 0
        self.auroraveils = [0,0]
        self.stealthrocks = [False, False]
        self.spikes = [0, 0]
        self.pokemon = [[], []]
        self.name2Indexes = [dict(), dict()]

    def loadPokemon(self, line):
        line = line.split("]")
        player = int(line[0].split("|")[2][1])-1
        line[0] = line[0][13:]
        for poke in line:
            pokemon = Pokemon(poke, pokemon_stats, move_dict)
            self.pokemon[player].append(pokemon)
