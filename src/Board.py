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
          self.active = [[None, None], [None, None]]
          self.justProtected = [[False, False], [False, False]]

     @staticmethod
     def switchArr(arr):
          temp = arr[0]
          arr[0] = arr[1]
          arr[1] = temp

     def switchSides(self):
          self.switchArr(self.tailwinds)
          self.switchArr(self.auroraveils)
          self.switchArr(self.lightscreens)
          self.switchArr(self.reflects)
          self.switchArr(self.stealthrocks)
          self.switchArr(self.spikes)
          self.switchArr(self.toxicspikes)
          self.switchArr(self.pokemon)
          self.switchArr(self.name2Indicies)
          self.switchArr(self.active)
          self.switchArr(self.justProtected)
          self.winner = 1 - self.winner


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
     
     @staticmethod
     def processName(name):
          name = name.replace(" ", "-")
          if name in pokemon_stats:
               return name
          else:
               names = name.split("-")
               if names[0] in pokemon_stats:
                    return names[0]
               else:
                    raise Exception(f"KEY ERROR: {name}")
          
     def j2pokemon(self, i, j):
          return self.active[i][j]

     def loadPokemon(self, line):
          line = line.split("]")
          player = int(line[0].split("|")[2][1])-1
          line[0] = line[0][13:]
          for i,poke in enumerate(line):
               name = Board.processName(poke.split("|")[0])
               pokemon = Pokemon(name, poke, pokemon_stats, move_dict)
               self.pokemon[player].append(pokemon)
               self.name2Indicies[player][name] = i
          if len(self.pokemon[player]) < 6:
               raise Exception("TOO FEW POKEMON")

     def switch(self, line):
          line = line.split("|")
          line[0] = line[0].split(" ")
          i,j = Board.player2indicies(line[0][0])

          # Switch out pokemon
          if self.active[i][j] is not None:
               k = self.j2pokemon(i, j)
               self.pokemon[i][k].switchOut()

          poke = Board.processName(line[1].split(",")[0])
          if poke in self.name2Indicies[i]:
               self.active[i][j] = self.name2Indicies[i][poke]
          else:
               # Alternate name
               poke = line[1].split(",")[0]
               self.active[i][j] = self.name2Indicies[i][poke]
          self.pokemon[i][self.active[i][j]].shown = True
          hp = int(line[2].split("/")[0])
          if hp != self.pokemon[i][self.active[i][j]].hp:
              print(f"Creo que amoonguss: {poke}")
          self.pokemon[i][self.active[i][j]].updateHp(hp)
         
     def startField(self, line):
          terrain = line.split("|")[0].split(":")[1][1:]
          if terrain == "Trick Room":
               self.trickroom = 5
          elif terrain == "Gravity":
               self.gravity = 5
          else:
               self.terrain[0] = terrain
               self.terrain[1] = 0

     def endField(self, line):
          terrain = line.split("|")[0].split(" ")[1]
          if terrain == "Trick Room":
               self.trickroom = 0
          elif terrain == "Gravity":
               self.gravity = 0
          else:
               self.terrain[0] = "none"
               self.terrain[1] = 0
         
     def startWeather(self, line):
          weather = line.split("|")[0]
          self.weather[0] = weather
          self.weather[1] = 0

     def updateSide(self, line, start):
          line = line.split("|")
          player, _ = Board.player2indicies(line[0])
          if line[1] == "Reflect":
               self.reflects[player] = [start, 0]
          else:
               splitted = line[1].split(":")
               if len(splitted) <= 1:
                    raise Exception(f"UNKNOWN EFFECT: {line[1]}")
               effect = splitted[1][1:]
               if effect == "Aurora Veil":
                    self.auroraveils[player] = [start, 0]
               elif effect == "Light Screen":
                    self.lightscreens[player] = [start, 0]  
               elif effect == "Tailwind":
                    if start:
                         self.tailwinds[player] = 4
               else:
                    raise Exception(f"UNKNOWN EFFECT: {effect}")

     def boost(self, line):
          line = line.split("|")
          i, j = Board.player2indicies(line[0])
          j = self.j2pokemon(i, j)
          stat = Board.stat2index(line[1])
          change = int(line[2])
          self.pokemon[i][j].updateBoost(stat, change)

     def unboost(self, line):
          line = line.split("|")
          i, j = Board.player2indicies(line[0])
          j = self.j2pokemon(i, j)
          stat = Board.stat2index(line[1])
          change = -int(line[2])
          self.pokemon[i][j].updateBoost(stat, change)

     def updateHP(self, line):
          line = line.split("|")
          i, j = Board.player2indicies(line[0])
          if line[0][2] == ":":
               # J value is garbage (must lookup manually)
               name = Board.processName(line[0].split(" ")[1])
               j = self.name2Indicies[i][name]
          else:
               j = self.j2pokemon(i, j)
          if line[1] == "0 fnt":
               self.pokemon[i][j].updateHp(0)
          else:
               hp = int(line[1].split("/")[0])
               self.pokemon[i][j].updateHp(hp)
              

     def endItem(self, line):
          line = line.split("|")
          i, j = Board.player2indicies(line[0])
          j = self.j2pokemon(i, j)
          self.pokemon[i][j].lostItem = True

     def tera(self, line):
          line = line.split("|")
          i, j = Board.player2indicies(line[0])
          j = self.j2pokemon(i, j)
          self.pokemon[i][j].tera = True

     def protected(self, line):
          line = line.split("|")
          i,k = Board.player2indicies(line[0])
          j = self.j2pokemon(i, k)
          self.pokemon[i][j].justProtected = True
          self.justProtected[i][k] = True

     def startSub(self, line):
          line = line.split("|")
          i,k = Board.player2indicies(line[0])
          j = self.j2pokemon(i, k)
          self.pokemon[i][j].sub = True

     def endSub(self, line):
          line = line.split("|")
          i,k = Board.player2indicies(line[0])
          j = self.j2pokemon(i, k)
          self.pokemon[i][j].sub = False

     def perish(self, line):
          line = line.split("|")
          i,k = Board.player2indicies(line[0])
          j = self.j2pokemon(i, k)
          perishIx = 4 - int(line[1][-1])
          self.pokemon[i][j].perish = perishIx

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
               for j in range(2):
                    k = self.j2pokemon(player, j)
                    if self.justProtected[player][j] and self.pokemon[player][k].justProtected:
                         self.pokemon[player][k].justProtected = False
          self.justProtected = [[False, False], [False, False]]