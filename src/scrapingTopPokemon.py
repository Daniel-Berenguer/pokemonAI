from bs4 import BeautifulSoup
import requests

URL = "https://munchstats.com/gen9vgc2025regi"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
results = soup.find(id="pokemon-list")
items = results.find_all("li")

N_POKEMON = 256

pokemon = []

for li in items[:N_POKEMON]:
    poke = li.find_all(class_="left-text")[0].text
    pokemon.append(poke)

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

moves = []
move_names = set([])

for poke in pokemon:
    url = f"https://munchstats.com/gen9vgc2025regi/1760/{poke}"
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")
    h2 = soup.find("h2", string=lambda text: text and "moves" in text.lower())
    moves_div = h2.find_parent("div")
    move_list = moves_div.find("ul")
    items = move_list.find_all("li")

    for item in items:
        name = item.find(class_="left-text").text
        if name not in move_names:
            move_names.add(name)
            btn_text = item.find(class_="export-button has-tooltip").get("data-tooltip").split("\n")

            # Get type and status/physical
            tipo, clase = btn_text[0].split(" ")
            clase = clase[1:-1]

            # Base power
            power = btn_text[1].split(" ")[2]
            if power == "N/A":
                power = 0
            else:
                power = int(power)

            # Accuracy
            accuracy = btn_text[2].split(" ")[1]
            if accuracy == "N/A":
                accuracy = 100
            else:
                accuracy = int(accuracy)

            # Priority
            priority = int(btn_text[3].split(" ")[1])

            move = Move(name, clase, power, accuracy, tipo, priority)
            moves.append(move)
            print(move)

import pickle

print(f"Nº pokemon: {len(pokemon)}")
print(f"Nº Moves: {len(moves)}")


with open("pokemon-name-list", "wb") as file:
    pickle.dump(pokemon, file)

with open("moves", "wb") as file:
    pickle.dump(moves, file)



        