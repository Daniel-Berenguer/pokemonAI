from Board import Board

with open("data/games/game", encoding="utf-8") as file:
    text = file.read()

lines = text.split("\n")

# i -> Line index

board = Board()

# First we get the winner
p1 = None
p2 = None
for line in lines:
    if line.startswith("|player|p1|"):
        p1 = line.split("|")[2]
    elif line.startswith("|player|p2|"):
        p2 = line.split("|")
    elif line.startswith("|win|"):
        winnerName = line.split("|")[1]
        if winnerName == p1:
            board.winner = 0
        else:
            board.winner = 1


for line in lines:
    if line.startswith("|showteam|"):
        board.loadPokemon(line)

    if line.startswith("|switch|"):
        board.switch(line.strip("|switch|"))

    if line.startswith("|-fieldstart|"):
        board.startField(line.strip("|-fieldstart|"))

    if line.startswith("|-sidestart|"):
        board.startSide(line.strip("|-sidestart|"))

    if line.startswith("|-weather|"):
        board.startWeather(line.strip("|-weather|"))

    if line.startswith("|-boost|"):
        board.boost(line.strip("|-boost|"))

    if line.startswith("|-unboost|"):
        board.boost(line.strip("|-unboost|"))

    if line.startswith("|-damage|"):
        board.updateHP(line.strip("|-damage|"))

    if line.startswith("|-heal|"):
        board.updateHP(line.strip("|-heal|"))

    if line.startswith("|-enditem|"):
        board.endItem(line.strip("|-enditem|"))

    if line.startswith("|-fieldend|"):
        board.endItem(line.strip("|-fieldend|"))

    if line.startswith("|-singleturn|") and "Protect" in line:
        board.protected(line.strip("|-singleturn|"))

for i, pokemons in enumerate(board.pokemon):
    print(f"Player: {i}")
    for poke in pokemons:
        print(poke)
