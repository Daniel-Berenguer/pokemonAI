from Board import Board

with open("data/games/game", encoding="utf-8") as file:
    text = file.read()

lines = text.split("\n")

# i -> Line index

board = Board()

for i,line in enumerate(lines):
    if line.startswith("|showteam|"):
        board.loadPokemon(line)

for i, pokemons in enumerate(board.pokemon):
    print(f"Player: {i}")
    for poke in pokemons:
        print(poke)
