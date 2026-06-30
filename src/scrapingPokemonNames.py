from bs4 import BeautifulSoup
import requests

URL = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_in_Pok%C3%A9mon_Champions"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
results = soup.find("table")
rows = results.find_all("tr")

pokes = []

for row in rows[1:]:
    cols = row.find_all("td")
    pokes.append(cols[1].text.strip("\n").lower())

print(pokes)