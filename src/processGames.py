from Board import Board


def processGame(filename):
    with open(f"data/games/{filename}", encoding="utf-8") as file:
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
            board.switch(line[len("|switch|"):])

        if line.startswith("|-fieldstart|"):
            board.startField(line[len("|-fieldstart|"):])

        if line.startswith("|-sidestart|"):
            board.updateSide(line[len("|-sidestart|"):], True)

        if line.startswith("|-sideend|"):
            board.updateSide(line[len("|-sideend|"):], False)

        if line.startswith("|-weather|"):
            board.startWeather(line[len("|-weather|"):])

        if line.startswith("|-boost|"):
            board.boost(line[len("|-boost|"):])

        if line.startswith("|-unboost|"):
            board.boost(line[len("|-unboost|"):])

        if line.startswith("|-damage|"):
            board.updateHP(line[len("|-damage|"):])

        if line.startswith("|-heal|"):
            board.updateHP(line[len("|-heal|"):])

        if line.startswith("|-enditem|"):
            board.endItem(line[len("|-enditem|"):])

        if line.startswith("|-fieldend|"):
            board.endItem(line[len("|-fieldend|"):])

        if line.startswith("|-singleturn|") and "Protect" in line:
            board.protected(line[len("|-singleturn|"):])

        if line.startswith("-terastallize"):
            board.tera(line[len("-terastallize"):])

        if line.startswith("|-start|") and "Substitute" in line:
            board.startSub(line[len("|-singleturn|"):])

        if line.startswith("|-end|") and "Substitute" in line:
            board.endSub(line[len("|-singleturn|"):])

        if line.startswith("|turn|"):
            board.nextTurn()

import os
i = 1
for filename in os.listdir("data/games")[:80]:
    print(i)
    i += 1
    try:
        processGame(filename)
    except Exception as e:
        print(e)