from bs4 import BeautifulSoup
import requests

pokemonNames = []

with open("data/pokemon.csv", "r") as file:
    for line in file.read().split("\n"):
        pokemonNames.append(line)


missing_pokemon = []
pokemon = {}

for name in pokemonNames:
    url = f"https://pokemondb.net/pokedex/{name}"
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    h2 = soup.find("h2", string=lambda text: text and "base stats" in text.lower())

    if h2 is None:
        splitted_name = name.split("-")
        first = splitted_name[0]
        second = splitted_name[1]

        url = f"https://pokemondb.net/pokedex/{first}"
        page = requests.get(url)

        soup = BeautifulSoup(page.content, "html.parser")

        TABS = {"raichu-alola" : 1, "slowbro-galar" : 2, "floette-eternal" : 1, "floette-mega" : 2, "rotom-heat" : 1, "rotom-wash" : 2,
                "rotom-frost" : 3, "rotom-fan" : 4, "rotom-mow" : 5, "tauros-paldea-combat" : 1, "tauros-paldea-aqua" : 3,
                "tauros-paldea-blaze" : 2, "slowbro-mega" : 1, "absol-mega" : 1, "garchomp-mega" : 1, "lucario-mega" : 1,
                "greninja-mega" : 2, "meowstic-mega" : 2, "charizard-mega-x" : 1, "charizard-mega-y" : 2,
                "raichu-mega-x" : 2, "raichu-mega-y" : 3}

        tab_list = soup.find("div", class_="sv-tabs-panel-list")
        tabs = tab_list.find_all(recursive=False)

        if name in TABS:
            soup = tabs[TABS[name]]
            h2 = soup.find("h2", string=lambda text: text and "base stats" in text.lower())
        elif second in ["hisui", "alola", "galar", "paldea", "mega"] and len(tabs) == 2:
            soup = tabs[1]
            h2 = soup.find("h2", string=lambda text: text and "base stats" in text.lower())

    if h2 is None:
        missing_pokemon.append(name)

    else:
        moves_div = h2.find_parent("div")
        table = moves_div.find(class_="vitals-table")
        body = table.find("tbody")
        trs = body.find_all("tr")

        stats = []

        for tr in trs:
            td = tr.find_all("td")[0]
            stats.append(td.text)

        th = soup.find("th", string=lambda text: text and "Type" in text)
        type_tr = th.find_parent("tr")
        td = type_tr.find("td")
        a = td.find_all("a")

        type1 = a[0].text
        if len(a) > 1:
            type2 = a[1].text
        else:
            type2 = "None"

        stats = [type1, type2] + stats

        try:
            if "mega" in name and name != "meganium":
                th = soup.find("th", string=lambda text: text and "Abilities" in text)
                ab_tr = th.find_parent("tr")
                td = ab_tr.find("td")
                a = td.find("a")
                ab = a.text
                stats.append(ab.lower())

        except Exception as e:
            print(e)

        print(name, stats)

        pokemon[name] = stats

print(f"Nº Pokemon: {len(pokemon)}")
print("Missing: ")
print(missing_pokemon)

with open("data/pokemon-stats.csv", 'w') as file:
    text = ""
    for poke in pokemon:
        text += poke
        for stat in pokemon[poke]:
            text += "," + stat
        text += "\n"
    file.write(text.strip("\n"))