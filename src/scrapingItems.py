from bs4 import BeautifulSoup
import requests
import pickle

with open("pokemon-name-list", "rb") as file:
    pokemon = pickle.load(file)

items = set([])

for poke in pokemon:
    url = f"https://munchstats.com/gen9vgc2025regi/1760/{poke}"
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")
    h2 = soup.find("h2", string=lambda text: text and "items" in text.lower())
    items_div = h2.find_parent("div")
    item_list = items_div.find("ul")
    lis = item_list.find_all("li")

    for li in lis:
        name = li.find(class_="left-text").text
        if name not in items:
            items.add(name)
            print(name)


print(len(items))

with open("items", "wb") as file:
    pickle.dump(items, file)