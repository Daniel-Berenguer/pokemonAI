import pickle
import csv


class Move:
    def __init__(self, name, clase, power, accuracy, type, priority):
        self.name = name
        self.clase = clase
        self.power = power
        self.accuracy = accuracy
        self.type = type
        self.priority = priority

    def __str__(self):
        return self.name + ","  + self.type + ","  + self.clase + ","  + str(self.power) + ","  + str(self.accuracy) + ","  + str(self.priority)

with open("abilities", "rb") as file:
    abilities = pickle.load(file)

with open("data/abilities.csv", 'w') as file:
    text = ""
    for ab in abilities:
        text += ab + "\n"
    file.write(text.replace("(", "").replace(")", "").replace(" ", ""))

with open("items", "rb") as file:
    items = pickle.load(file)

with open("data/items.csv", 'w') as file:
    text = ""
    for item in items:
        text += item + "\n"
    file.write(text.replace(" ", ""))

    
with open("moves", "rb") as file:
    moves = pickle.load(file)

with open("data/moves.csv", 'w') as file:
    text = ""
    move_dict = dict([])
    for move in moves:
        text += move.__str__() + "\n"

    
        a = move.__str__().replace(" ", "").split(",")
        if a[0] != "Nothing":
            move_dict[a[0]] = a[1:]

    file.write(text.replace(" ", ""))

with open("data/moves-dict", "wb") as file:
    pickle.dump(move_dict, file)

with open("pokemon-name-list", "rb") as file:
    pokemon = pickle.load(file)

with open("data/pokemon.csv", 'w') as file:
    text = ""
    for poke in pokemon:
        text += poke + "\n"
    file.write(text)


with open("pokemon", "rb") as file:
    pokemon = pickle.load(file)

with open("data/pokemon-stats.csv", 'w') as file:
    text = ""
    for poke in pokemon:
        text += poke
        for stat in pokemon[poke]:
            text += "," + stat
        text += "\n"
    file.write(text)