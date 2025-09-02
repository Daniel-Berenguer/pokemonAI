from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time
import os

existing_games = os.listdir("data/games/")

new = 0

for p in range(1,35):
    print("Page number: ", p)
    options = Options()
    options.add_argument("--headless=new") 
    driver = webdriver.Chrome(options=options)
    options.add_argument("--headless=new")
    
    driver.get(f"https://replay.pokemonshowdown.com/?format=%5BGen%209%5D%20VGC%202025%20Reg%20H%20(Bo3)&page={p}")
    #driver.get(f"https://replay.pokemonshowdown.com/?format=%5BGen%209%5D%20VGC%202025%20Reg%20H%20(Bo3)&page={p}&sort=rating")

    time.sleep(3)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    links = []

    ul = soup.find("ul", class_="linklist")
    lis = ul.find_all("li")
    for li in lis:
        a = li.find("a")
        links.append(a.get("href"))

    for link in links:
        if link not in existing_games:
            new += 1
            url = f"https://replay.pokemonshowdown.com/{link}.log"
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            print(link)
            with open(f"data/games/{link}", 'w', encoding="utf-8") as file:
                file.write(soup.text)
        else:
            print("repe")


print(f"New games: {new}")