class Pokemon:
    def __init__(self, text, pokemon_stats, move_dict):
        text = text.split("|")
        self.name = text[0].replace(" ", "-")
        self.item = text[2]
        self.ability = text[3]
        self.moves = []
        self.stats = pokemon_stats[self.name]
        moveNames = text[4].split(",")
        for moveName in moveNames:
            self.moves.append([moveName] + move_dict[moveName])
        self.teratype = text[11].strip(",")
        self.boosts = [0, 0, 0, 0, 0]
        self.status = "None"
        self.hp = 100
        self.tera = False
        self.saltCure = False


    def __str__(self):
        movetxt = ""
        for move in self.moves:
            movetxt += "," + move[0]
        return self.name + "," + self.item + "," + self.ability + "," +  self.teratype + movetxt
        

if __name__ == "__main__":
    text = "Calyrex-Ice||Leftovers|AsOneGlastrier|GlacialLance,LeechSeed,TrickRoom,Protect||||||50|,,,,,Fairy"
    import pickle
    with open("pokemon", "rb") as file:
        pokemon_stats = pickle.load(file)
    with open("data/moves-dict", "rb") as file:
        move_dict = pickle.load(file)

    poke = Pokemon(text, pokemon_stats, move_dict)
    print(poke)