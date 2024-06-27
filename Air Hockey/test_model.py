import pygame
import torch
from game_core import GameCore
from gui_core import GUICore


def test_model(policy_net, mode='player_vs_bot'):
    pygame.init()
    board_image = pygame.image.load("assets/board.png")
    board_width, board_height = board_image.get_rect().size
    screen = pygame.display.set_mode((board_width, board_height))
    pygame.display.set_caption("Air Hockey Test")

    gui = GUICore(screen, board_image, board_width, board_height)
    game = GameCore(gui, board_width, board_height)
    game.reset_game()

    state = game.get_state()
    state = torch.tensor([state], dtype=torch.float32)

    while game.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
            elif event.type == pygame.MOUSEMOTION and mode == 'player_vs_bot':
                mouse_x, mouse_y = pygame.mouse.get_pos()
                game.move_paddle(1, mouse_x, mouse_y)

        if mode == 'player_vs_bot':
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1).item()
            game.take_action(action, 2)  # Player 2 is the bot

        elif mode == 'bot_vs_bot':
            with torch.no_grad():
                action1 = policy_net(state).max(1)[1].view(1, 1).item()
                action2 = policy_net(state).max(1)[1].view(1, 1).item()
            game.take_action(action1, 1)  # Player 1 is the bot
            game.take_action(action2, 2)  # Player 2 is the bot

        next_state = game.get_state()
        next_state = torch.tensor([next_state], dtype=torch.float32)

        state = next_state

        gui.update(game.paddle1_pos, game.paddle2_pos, game.puck_pos, game.goals)
        pygame.time.delay(30)

    pygame.quit()
    print("Game Over")
