from bs4 import BeautifulSoup
import requests
import pickle

with open("pokemon-name-list", "rb") as file:
    pokemon = pickle.load(file)

abilities = set([])

for poke in pokemon:
    url = f"https://munchstats.com/gen9vgc2025regi/1760/{poke}"
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")
    h2 = soup.find("h2", string=lambda text: text and "abilities" in text.lower())
    abs_div = h2.find_parent("div")
    abs_list = abs_div.find("ul")
    lis = abs_list.find_all("li")

    for li in lis:
        name = li.find(class_="left-text").text
        if name not in abilities:
            abilities.add(name)
            print(name)

print("--------------------------------------------------")
print(abilities)
print(len(abilities))


with open("abilities", "wb") as file:
    pickle.dump(abilities, file)