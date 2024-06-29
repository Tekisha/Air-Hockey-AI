import pygame
from pygame.locals import *


class GUICore:
    def __init__(self, screen, board_image, board_width, board_height):
        self.screen = screen
        self.board_image = board_image
        self.board_rect = self.board_image.get_rect()
        self.board_width = board_width
        self.board_height = board_height

    def update(self, paddle1_pos, paddle2_pos, puck_pos, goals, predicted_path):
        self.screen.blit(self.board_image, self.board_rect)

        # Draw paddles and puck
        pygame.draw.circle(
            self.screen, (0, 0, 255), (paddle1_pos["x"], paddle1_pos["y"]), 20
        )
        pygame.draw.circle(
            self.screen, (255, 0, 0), (paddle2_pos["x"], paddle2_pos["y"]), 20
        )
        pygame.draw.circle(self.screen, (0, 0, 0), (puck_pos["x"], puck_pos["y"]), 10)

        # Draw scores
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"{goals['left']} - {goals['right']}", True, (0, 0, 0))
        self.screen.blit(
            score_text,
            (
                self.board_width // 2 - score_text.get_width() // 2,
                self.board_height - 30,
            ),
        )

        # Draw player names
        player1_name = font.render("Player 1", True, (0, 0, 255))
        player2_name = font.render("Player 2", True, (255, 0, 0))
        self.screen.blit(player1_name, (50, self.board_height - 30))
        self.screen.blit(
            player2_name,
            (self.board_width - player2_name.get_width() - 50, self.board_height - 30),
        )

        self.draw_predicted_path(predicted_path)

        pygame.display.flip()

    def close(self):
        pygame.quit()

    def draw_predicted_path(self, predicted_path):
        for position in predicted_path:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(position["x"]), int(position["y"])), 3)
