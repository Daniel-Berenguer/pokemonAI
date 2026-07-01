from bs4 import BeautifulSoup
import requests

URL = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_in_Pok%C3%A9mon_Champions"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
results = soup.find_all("table")
poke_table = results[0]
mega_table = results[1]

pokes = []

REGION_DEMONYMS = ["alolan", "hisuian", "galarian"]
REGIONS = ["alola", "hisui", "galar"]

rows = poke_table.find_all("tr")

for row in rows[1:]:
    cols = row.find_all("td")
    name = cols[1].text.lower().strip("\n")
    if "form" in name:
        for i,region in enumerate(REGION_DEMONYMS):
            if region in name:
                name = name.replace(region + " form", "")
                name = name + "-" + REGIONS[i]

        if "tauros" in name:
            continue

    if "floette" in name:
        name = "floette-eternal"

    pokes.append(name)

pokes.append("rotom-heat")
pokes.append("rotom-wash")
pokes.append("rotom-mow")
pokes.append("rotom-frost")
pokes.append("rotom-fan")
pokes.append("tauros-paldea-combat")
pokes.append("tauros-paldea-aqua")
pokes.append("tauros-paldea-blaze")

rows = mega_table.find_all("tr")

for row in rows[1:]:
    cols = row.find_all("td")
    name = cols[1].text.split(" ")[0].lower().replace("mega", "")
    if name not in ["raichu", "charizard", "nium"]:
        pokes.append(name + "-mega")

pokes.append("meganium-mega")
pokes.append("charizard-mega-x")
pokes.append("charizard-mega-y")
pokes.append("raichu-mega-x")
pokes.append("raichu-mega-y")

with open("data/pokemon.csv", "w") as file:
    text = ""
    for poke in pokes:
        text += poke + "\n"
    file.write(text.strip("\n"))

print(len(pokes))