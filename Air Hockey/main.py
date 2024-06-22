import pygame
from game_core import GameCore
from gui_core import GUICore


def main():
    pygame.init()
    board_image = pygame.image.load("assets/board.png")
    board_width, board_height = board_image.get_rect().size

    screen = pygame.display.set_mode((board_width, board_height))
    pygame.display.set_caption("Air Hockey")

    gui = GUICore(screen, board_image, board_width, board_height)
    game = GameCore(gui, board_width, board_height)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                game.move_paddle(1, mouse_x, mouse_y)

        game.update_game_state()
        gui.update(game.paddle1_pos, game.paddle2_pos, game.puck_pos, game.goals)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
