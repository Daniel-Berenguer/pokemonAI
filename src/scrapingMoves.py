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
    
driver.get("https://game8.co/games/Pokemon-Champions/archives/590397")
time.sleep(3)

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

results = soup.find_all("table")[1]
rows = results.find_all("tr")

moves = []

for i,row in enumerate(rows):
    if i % 2 == 1:
        move = []
        cols = row.find_all("td")
        
        # Move Name
        move.append(cols[0].text[:-1].lower())

        # Type
        move.append(cols[1].find("img")["alt"].split(" ")[0].lower())

        # Category
        move.append(cols[2].find("img")["alt"].split(" ")[0].lower())

        # Power
        power = cols[3].text
        if power == "-":
            power = "0"
        move.append(power)

        # Accuracy
        acc = cols[4].text
        if acc == "-":
            acc = "100"
        move.append(acc)

        moves.append(move)

with open("data/moves.csv", "w") as file:
    text = ""
    for move in moves:
        for col in move:
            text += col + ","
        text = text[:-1] + "\n"
    file.write(text.strip("\n"))

print(len(moves))
