import pygame

class Lib:
    def __init__(self, game):
        pygame.init()
        self.square_size = 15
        pygame.font.init()
        pygame.display.set_caption('MyGame')
        self.screen = pygame.display.set_mode((game.board_width * self.square_size, game.board_height * self.square_size))
        self.game = game

    def __del__(self):
        pygame.quit()

    """
    def display_ui(self):
        myfont = pygame.font.SysFont('Segoe UI', 20)
        myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
        text_score = myfont.render('SCORE: ', True, (0, 0, 0))

        text_score_number = myfont.render(str(score), True, (0, 0, 0))
        text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
        text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
        self.screen.blit(text_score, (45, 440))
        self.screen.blit(text_score_number, (120, 440))
        self.screen.blit(text_highest, (190, 440))
        self.screen.blit(text_highest_number, (350, 440))
        self.screen.blit(self.bg, (10, 10))
    """

    def display_board(self):
        food_size = 1
        stone_size = 0.5
        stone_offset = self.square_size / 2 - stone_size * self.square_size / 2
        for y in range(len(self.game.board)):
            for x in range(len(self.game.board[y])):
                if (self.game.board[y][x].food):
                    pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(y * self.square_size, x * self.square_size, self.square_size * food_size, self.square_size * food_size))
                if (self.game.board[y][x].stone):
                    pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(y * self.square_size + stone_offset, x * self.square_size + stone_offset, self.square_size * stone_size, int(self.square_size * stone_size)))

    def display_players(self):
        for player in self.game.players:
            if player.dead is False:
                pygame.draw.rect(self.screen, player.color, pygame.Rect(player.y * self.square_size, player.x * self.square_size, self.square_size, self.square_size))

    def display_player_vision(self):
        for player in self.game.players:
            if player.dead is True:
                continue
            vision = player.take_a_look()
            for square_index in range(len(vision)):
                pygame.draw.rect(self.screen, (player.color[0] / 2 % 255, player.color[1] / 2 % 255, player.color[2] / 2), pygame.Rect(vision[square_index].y * self.square_size, vision[square_index].x * self.square_size, self.square_size, self.square_size))

    def update(self):
        self.screen.fill((255, 255, 255))
        self.display_player_vision()
        self.display_players()
        self.display_board()
        # self.display_ui(game)
        pygame.display.update()
        for player in self.game.players:
            if player.agent.reward != 0:
                self.game.turn_latency = self.game.highlight_turn_latency
                break
            self.game.turn_latency = self.game.general_turn_latency
        pygame.time.wait(self.game.turn_latency)
