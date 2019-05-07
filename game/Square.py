class Square:
    def __init__(self, y_loc, x_loc):
        self.x = x_loc
        self.y = y_loc
        self.food = False
        self.trap = False
        self.stone = False
        self.players = []
