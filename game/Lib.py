import pygame
from pygame.locals import *

class Lib:
    def __init__(self, game):
        pygame.init()
        self.square_size = 15
        pygame.font.init()
        self.game = game
        pygame.display.set_caption(f'Bear Grylls Project: game {self.game.id}')
        self.screen = pygame.display.set_mode((game.board_width * self.square_size, game.board_height * self.square_size + 10 * self.square_size))

    def __del__(self):
        pygame.quit()

    def display_ui(self):
        myfont = pygame.font.SysFont('Segoe UI', 20)
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, self.game.board_height * self.square_size, self.square_size * self.game.board_width, self.square_size * 2))
        self.screen.blit(myfont.render(f"food to win : {self.game.max_score}", True, (255, 255, 255)), (self.square_size * 3, self.game.board_height * self.square_size))
        self.screen.blit(myfont.render(f"max turns : {self.game.max_turns}", True, (255, 255, 255)), (self.square_size * 20, self.game.board_height * self.square_size))
        self.screen.blit(myfont.render(f"current turn index : {len(self.game.players[0].food_scores)}", True, (255, 255, 255)), (self.square_size * 30, self.game.board_height * self.square_size))
        self.screen.blit(myfont.render(f"max food on board : {self.game.food_offset}", True, (255, 255, 255)), (self.square_size * 3, (self.game.board_height + 1) * self.square_size))
        self.screen.blit(myfont.render(f"current turn latency : {self.game.turn_latency}", True, (255, 255, 255)), (self.square_size * 20, (self.game.board_height + 1) * self.square_size))
        y = (2 + self.game.board_height) * self.square_size
        self.screen.blit(myfont.render(f"player name", True, (0, 0, 0)), (self.square_size * 3, y))
        self.screen.blit(myfont.render(f"survival time", True, (0, 0, 0)), (self.square_size * 10, y))
        self.screen.blit(myfont.render(f"food", True, (0, 0, 0)), (self.square_size * 17, y))
        self.screen.blit(myfont.render(f"stones", True, (0, 0, 0)), (self.square_size * 20, y))
        self.screen.blit(myfont.render(f"death count", True, (0, 0, 0)), (self.square_size * 25, y))
        self.screen.blit(myfont.render(f"max score", True, (0, 0, 0)), (self.square_size * 30, y))
        self.screen.blit(myfont.render(f"max survival time", True, (0, 0, 0)), (self.square_size * 35, y))
        for player in self.game.players:
            y = (5 + self.game.players.index(player) * 2 + self.game.board_height) * self.square_size
            pygame.draw.rect(self.screen, player.color, pygame.Rect(self.square_size, y, self.square_size, self.square_size))
            self.screen.blit(myfont.render(f"{player.name}", True, (0, 0, 0)), (self.square_size * 3, y))
            self.screen.blit(myfont.render(f"{player.survival_time}", True, (0, 0, 0)), (self.square_size * 10, y))
            self.screen.blit(myfont.render(f"{str(player.food)[0:4]}", True, (0, 0, 0)), (self.square_size * 17, y))
            self.screen.blit(myfont.render(f"{player.stones}", True, (0, 0, 0)), (self.square_size * 20, y))
            self.screen.blit(myfont.render(f"{len(player.death_counter)}", True, (0, 0, 0)), (self.square_size * 25, y))
            self.screen.blit(myfont.render(f"{str(player.max_score_reached)[0:4]}", True, (0, 0, 0)), (self.square_size * 30, y))
            self.screen.blit(myfont.render(f"{player.max_survival_time}", True, (0, 0, 0)), (self.square_size * 35, y))

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
                if (self.game.board[y][x].trap):
                    pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(y * self.square_size, x * self.square_size, self.square_size * food_size, self.square_size * food_size))

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
        self.display_ui()
        # self.display_ui(game)
        pygame.display.update()
        """
        for player in self.game.players:
            if player.agent.reward != 0:
                self.game.turn_latency = self.game.highlight_turn_latency
                break
            self.game.turn_latency = self.game.general_turn_latency
        """
        pygame.time.wait(self.game.turn_latency)

    def get_key(self):
        pygame.event.pump()
        key = pygame.key.get_pressed()
        if key[K_q]:
            return 'q'
        if key[K_d]:
            return 'd'
        if key[K_f]:
            return 'f'
        if key[K_s]:
            return 's'
        if key[K_r]:
            return 'r'
        if key[K_p]:
            return 'p'
        return -1
