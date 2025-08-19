from bs4 import BeautifulSoup
import requests


url = f"https://replay.pokemonshowdown.com/gen9vgc2025regibo3-2379851730-f406uog6eeu2nu1kkirp4vxslvnnaaupw.log"
page = requests.get(url)

soup = BeautifulSoup(page.content, "html.parser")
text = soup.text

with open("data/games/game", "w", encoding="utf-8") as file:
    file.write(text)