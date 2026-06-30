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
    
driver.get("https://game8.co/games/Pokemon-Champions/archives/588871")
time.sleep(3)

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

results = soup.find_all("table")[1]
rows = results.find_all("tr")

items = []

for i,row in enumerate(rows[1:]):
    items.append(row.find("td").text.lower().replace(" ", "").replace("-",""))

with open("data/items.csv", "w") as file:
    text = ""
    for item in items:
        text += item + "\n"
    file.write(text.strip("\n"))

print(len(items))
