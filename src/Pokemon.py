class Pokemon:
    def __init__(self, name, player, text, pokemon_stats, move_dict):
        text = text.split("|")
        self.name = name.lower()
        self.item = text[2].lower()
        self.ability = text[3].lower()
        self.moves = []
        self.stats = pokemon_stats[self.name]
        moveNames = text[4].split(",")
        for moveName in moveNames:
            moveName = moveName.lower()
            self.moves.append([moveName] + move_dict[moveName])
        while len(self.moves) < 4:
            self.moves.append(["nothing"] + move_dict["nothing"])
        self.boosts = [0, 0, 0, 0, 0]
        self.status = "none"
        self.hp = 100
        self.saltCure = False
        self.justProtected = False
        self.sub = False
        self.lostItem = False
        self.fnt = False
        self.shown = False
        self.perish = 0
        self.status = "none"
        self.team = player
        self.mega_evolved = False

    def switchOut(self):
        self.perish = 0
        self.sub = False
        self.saltCure = False
        self.justProtected = False
        self.boosts = [0, 0, 0, 0, 0]

    def updateHp(self, hp):
        self.hp = hp
        if hp == 0:
            self.fnt = True
        else:
            self.fnt = False

    def updateBoost(self,stat, change):
        self.boosts[stat-1] += change

    def mega(self, meganame, pokemon_stats):
        stats = pokemon_stats[self.name]
        self.ability = stats[-1]
        self.stats = stats[:-1]
        self.mega_evolved = True


    def __str__(self):
        movetxt = ""
        for move in self.moves:
            movetxt += "," + move[0]
        return self.name + "," + self.item + "," + self.ability + "," + movetxt
        

if __name__ == "__main__":
    text = "Calyrex-Ice||Leftovers|AsOneGlastrier|GlacialLance,LeechSeed,TrickRoom,Protect||||||50|,,,,,Fairy"
    import pickle
    with open("pokemon", "rb") as file:
        pokemon_stats = pickle.load(file)
    with open("data/moves-dict", "rb") as file:
        move_dict = pickle.load(file)

    poke = Pokemon(text, pokemon_stats, move_dict)
    print(poke)