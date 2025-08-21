from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import time

for p in range(1,10):
    driver = webdriver.Chrome()
    driver.get(f"https://replay.pokemonshowdown.com/?format=%5BGen%209%5D%20VGC%202025%20Reg%20H%20(Bo3)&page={p}&sort=rating")

    time.sleep(2)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    print(soup.prettify)

    links = []

    ul = soup.find("ul", class_="linklist")
    lis = ul.find_all("li")
    for li in lis:
        a = li.find("a")
        links.append(a.get("href"))

    for link in links:
        url = f"https://replay.pokemonshowdown.com/{link}.log"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        with open(f"data/games/{link}", 'w', encoding="utf-8") as file:
            file.write(soup.text)


