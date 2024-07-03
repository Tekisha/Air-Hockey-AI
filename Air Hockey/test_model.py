import pygame
import torch
from agent import Agent_DDPG
from game_core import GameCore
from gui_core import GUICore
from model import Actor_Critic_Models


def test_model(state_dim, n_actions):
    pygame.init()
    board_image = pygame.image.load("assets/board.png")
    board_width, board_height = board_image.get_rect().size
    screen = pygame.display.set_mode((board_width, board_height))
    pygame.display.set_caption("Air Hockey Test - Player vs Bot")

    gui = GUICore(screen, board_image, board_width, board_height)
    game = GameCore(gui, board_width, board_height)

    model = Actor_Critic_Models(1, state_dim, n_actions)

    agent = Agent_DDPG(0, model, n_actions)

    agent.load_model()

    clock = pygame.time.Clock()

    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                game.move_paddle(1, mouse_x, mouse_y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    game.reset_game()

        state = game.get_state(2)
        action = agent.act(state)
        action = (action[0][0].item(), action[0][1].item())
        game.take_action(action, 2)
        game.update_game_state()
        game.check_for_goal()

        gui.update(
            game.paddle1_pos,
            game.paddle2_pos,
            game.puck_pos,
            game.goals,
        )
        pygame.display.flip()
        clock.tick(500)

    pygame.quit()
    print("Game Over")
