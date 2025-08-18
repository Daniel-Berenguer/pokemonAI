from bs4 import BeautifulSoup
import requests
import pickle

with open("pokemon-name-list", "rb") as file:
    pokemonNames = pickle.load(file)

missing_pokemon = []
pokemon = {}

for name in pokemonNames:
    name = name.replace(" ", "-")
    url = f"https://pokemondb.net/pokedex/{name}"
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    h2 = soup.find("h2", string=lambda text: text and "base stats" in text.lower())

    if h2 is None:
        splitted_name = name.split("-")
        first = splitted_name[0]
        second = splitted_name[1]

        name = name.replace(" ", "-")
        url = f"https://pokemondb.net/pokedex/{first}"
        page = requests.get(url)

        soup = BeautifulSoup(page.content, "html.parser")

        TABS = {"Calyrex-Shadow" : 2, "Calyrex-Ice" : 1, "Urshifu-Rapid-Strike" : 1, "Urshifu-Single-Strike" : 0,
                "Zamazenta-Crowned" : 1, "Zacian-Crowned" : 1, "Ogerpon-Wellspring" : 1, "Ogerpon-Hearthflame" : 2,
                "Ogerpon-Cornerstone" : 3, "Indeedee-M" : 0, "Indeedee-F" : 1, "Ursaluna-Bloodmoon" : 1, "Necrozma-Dawn-Wings" : 2,
                "Necrozma-Dusk-Mane" : 1, "Kyurem-White" : 1, "Kyurem-Black" : 2
                }

        tab_list = soup.find("div", class_="sv-tabs-panel-list")
        tabs = tab_list.find_all(recursive=False)

        if name in TABS:
            soup = tabs[TABS[name]]
            h2 = soup.find("h2", string=lambda text: text and "base stats" in text.lower())
        elif second in ["Hisui", "Alola", "Galar", "Origin", "Therian"] and len(tabs) == 2:
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

        print(name, stats)

        pokemon[name] = stats

print(f"Nº Pokemon: {len(pokemon)}")
print("Missing: ")
print(missing_pokemon)

with open("pokemon", "wb") as file:
    pickle.dump(pokemon, file)