from bs4 import BeautifulSoup
import requests

URL = "https://munchstats.com/gen9vgc2024regh/0"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
results = soup.find(id="pokemon-list")
items = results.find_all("li")

N_POKEMON = 256

pokemon = []

for li in items[:N_POKEMON]:
    poke = li.find_all(class_="left-text")[0].text
    pokemon.append(poke)

moves = []
move_names = set([])
abilities = set([])
items = set([])

for poke in pokemon:
    print(poke)
    url = f"https://munchstats.com/gen9vgc2024regh/1760/{poke}"
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")
    h2 = soup.find("h2", string=lambda text: text and "moves" in text.lower())
    moves_div = h2.find_parent("div")
    move_list = moves_div.find("ul")
    lis = move_list.find_all("li")

    for item in lis:
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
                power = "0"
            else:
                power = power

            # Accuracy
            accuracy = btn_text[2].split(" ")[1]
            if accuracy == "N/A":
                accuracy = "100"
            else:
                accuracy = accuracy

            # Priority
            priority = btn_text[3].split(" ")[1]

            atrs = [tipo, clase, power, accuracy, priority]
            txt = name.replace("-", "")
            for at in atrs:
                txt += "," + at
            moves.append(txt)

    h2 = soup.find("h2", string=lambda text: text and "abilities" in text.lower())
    abs_div = h2.find_parent("div")
    abs_list = abs_div.find("ul")
    lis = abs_list.find_all("li")

    for li in lis:
        name = li.find(class_="left-text").text
        if name not in abilities:
            abilities.add(name)

    h2 = soup.find("h2", string=lambda text: text and "items" in text.lower())
    items_div = h2.find_parent("div")
    item_list = items_div.find("ul")
    lis = item_list.find_all("li")

    for li in lis:
        name = li.find(class_="left-text").text
        if name not in items:
            items.add(name)



print(f"Nº pokemon: {len(pokemon)}")
print(f"Nº Moves: {len(moves)}")


with open("data/pokemon.csv", "w") as file:
    text = ""
    for poke in pokemon:
        text += poke + "\n"
    file.write(text.strip("\n"))

with open("data/items.csv", "w") as file:
    text = ""
    for item in items:
        text += item.replace("-", "") + "\n"
    file.write(text.replace(" ", "").strip("\n"))

with open("data/abilities.csv", 'w') as file:
    text = ""
    for ab in abilities:
        text += ab + "\n"
    file.write(text.replace("(", "").replace(")", "").replace(" ", "").replace("-", "").strip("\n"))


with open("data/moves.csv", 'w') as file:
    text = ""
    for move in moves:
        text += move + "\n"

    file.write(text.replace(" ", "").strip("\n"))

        