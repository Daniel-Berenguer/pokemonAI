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
        self.weather = ["none", 0]
        self.terrain = ["none", 0]
        self.trickroom = 0
        self.gravity = 0
        self.auroraveils = [[False, 0], [False, 0]]
        self.lightscreens = [[False, 0], [False, 0]]
        self.reflects = [[False, 0], [False, 0]]
        self.stealthrocks = [False, False]
        self.spikes = [0, 0]
        self.toxicspikes = [0, 0]
        self.pokemon = [[], []]
        self.name2Indicies = [dict(), dict()]
        self.winner = None
        self.shown = [[False]*6, [False]*6]
        self.active = [[None, None], [None, None]]

    @staticmethod
    def player2indicies(word):
        if word[1] == "1":
            i = 0
        else:
            i = 1
        if word[2] == "a":
            j = 0
        else:
            j = 1
        return i, j
    
    @staticmethod
    def stat2index(stat):
         if stat == "hp":
              return 0
         elif stat == "atk":
              return 1
         elif stat == "def":
              return 2
         elif stat == "spa":
              return 3
         elif stat == "spd":
              return 4
         else:
              return 5

    def loadPokemon(self, line):
        line = line.split("]")
        player = int(line[0].split("|")[2][1])-1
        line[0] = line[0][13:]
        for i,poke in enumerate(line):
            pokemon = Pokemon(poke, pokemon_stats, move_dict)
            self.pokemon[player].append(pokemon)
            self.name2Indicies[player][pokemon.name] = i

    def switch(self, line):
         line = line.split("|")
         line[0] = line[0].split(" ")
         print(line)
         i,j = Board.player2indicies(line[0][0])
         poke = line[0][1]
         if poke in self.name2Indicies[i]:
            self.active[i][j] = self.name2Indicies[i][poke]
         else:
            # Alternate name
            poke = line[1].split(",")[0]
            self.active[i][j] = self.name2Indicies[i][poke]
         health = int(line[2].split("/")[0])
         """ if health != self.pokemon[i][self.active[i][j]].hp:
              print("ALGO RARO PASA AQUI")
              print(poke)
              print(health)
              print(self.pokemon[i][self.active[i][j]].hp) """
         
    def startField(self, line):
         terrain = line.split("|")[0].split(" ")[1]
         self.terrain[0] = terrain
         self.terrain[1] = 0

    def endField(self, line):
         terrain = line.split("|")[0].split(" ")[1]
         self.terrain[0] = "none"
         self.terrain[1] = 0
         
    def startWeather(self, line):
         weather = line.split("|")[0]
         self.weather[0] = weather
         self.weather[1] = 0

    def startSide(self, line):
         line = line.split("|")
         player, _ = Board.player2indicies(line[0].split(" "))
         effect = line[1].split(":")[1:]
         if effect == "Aurora Veil":
              self.auroraveils[player] = [True, 0]
         elif effect == "Light Screen":
              self.lightscreens[player] = [True, 0]
         elif effect == "Reflect":
              self.reflects[player] = [True, 0]
         else:
              print(f"UNKNOWN EFFECT: {effect}")

    def boost(self, line):
         line = line.split("|")
         i, j = Board.player2indicies(line[0])
         stat = Board.stat2index(line[1])
         change = int(line[2])
         self.pokemon[i][j].boosts[stat] += change

    def unboost(self, line):
         line = line.split("|")
         i, j = Board.player2indicies(line[0])
         stat = Board.stat2index(line[1])
         change = int(line[2])
         self.pokemon[i][j].boosts[stat] -= change

    def updateHP(self, line):
         line = line.split("|")
         i, j = Board.player2indicies(line[0])
         if line[1] == "0 fnt":
            self.pokemon[i][j].fnt = True
         else:
            hp = int(line[1].split("/")[0])
            self.pokemon[i][j].hp = hp
              

    def endItem(self, line):
         line = line.split("|")
         i, j = Board.player2indicies(line[0])
         self.pokemon[i][j].lostItem = True

    def protected(self, line):
         line = line.split("|")
         i,j = Board.player2indicies(line[0])

    def nextTurn(self):
        if self.weather[0] != "none":
            self.weather[1] += 1
        if self.terrain[0] != "none":
            self.terrain[1] += 1
        if self.trickroom > 0:
            self.trickroom -= 1
        if self.gravity > 0:
            self.gravity -= 1
        for player in range(2):
            if self.auroraveils[player][0]:
                self.auroraveils[player][1] += 1
            if self.lightscreens[player][0]:
                self.lightscreens[player][1] += 1
            if self.reflects[player][0]:
                self.reflects[player][1] += 1
            if self.tailwinds[player] > 0:
                 self.tailwinds[player] -= 1