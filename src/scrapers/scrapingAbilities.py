from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time
import os


options = Options()
options.add_argument("--headless=new") 
driver = webdriver.Chrome(options=options)
options.add_argument("--headless=new")
    
driver.get("https://game8.co/games/Pokemon-Champions/archives/590403")
time.sleep(3)

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

results = soup.find_all("table")

abilities = []

for table in results[2:26]:
    rows = table.find_all("tr")
    for row in rows[1:]:
        abilities.append(row.find("td").text.lower().strip("\n"))

with open("data/abilities.csv", "w") as file:
    text = ""
    for ab in abilities:
        text += ab + "\n"
    file.write(text.strip("\n"))

print(len(abilities))
