import pygame
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from dqn import DQN, ReplayBuffer, select_action, train
from game_core import GameCore
from gui_core import GUICore


# def train_dqn(state_dim, n_actions):
#     # Initialize the environment
#     pygame.init()
#     board_image = pygame.image.load("assets/board.png")
#     board_width, board_height = board_image.get_rect().size
#     screen = pygame.display.set_mode((board_width, board_height))
#     pygame.display.set_caption("Air Hockey Training")
#
#     gui = GUICore(screen, board_image, board_width, board_height)
#     game = GameCore(gui, board_width, board_height)
#
#     # Define hyperparameters
#     n_episodes = 5000
#     max_steps = 10000
#     gamma = 0.99
#     epsilon_start = 1.0
#     epsilon_end = 0.1
#     epsilon_decay = 500
#     target_update = 10
#     memory_capacity = 10000
#     batch_size = 128
#
#     # Initialize networks and memory
#     policy_net = DQN(state_dim, n_actions)
#     target_net = DQN(state_dim, n_actions)
#     policy_net.load_state_dict(torch.load("policy_net.pth"))
#     target_net.load_state_dict(torch.load("target_net.pth"))
#     target_net.eval()
#
#     optimizer = optim.Adam(policy_net.parameters())
#     memory = ReplayBuffer(memory_capacity)
#
#     steps_done = 0
#     clock = pygame.time.Clock()
#
#     for episode in range(n_episodes):
#         game.reset_game()
#         state = game.get_state()
#         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Correct tensor conversion
#
#         for t in range(max_steps):
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     return
#                 elif event.type == pygame.MOUSEMOTION:
#                     mouse_x, mouse_y = pygame.mouse.get_pos()
#                     game.move_paddle(1, mouse_x, mouse_y)  # Player controls paddle 1
#
#             action = select_action(state, policy_net, steps_done, epsilon_end, epsilon_start, epsilon_decay, n_actions)
#             steps_done += 1
#             game.take_action(action.item(), 2)  # AI controls paddle 2
#
#             game.update_game_state()
#
#             next_state = game.get_state()
#             next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
#
#             reward = game.get_reward()
#             reward = torch.tensor([reward], dtype=torch.float32)
#
#             done = not game.running
#             done = torch.tensor([done], dtype=torch.float32)
#
#             memory.push(state, action, reward, next_state, done)
#
#             state = next_state
#
#             optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma)
#
#             gui.update(game.paddle1_pos, game.paddle2_pos, game.puck_pos, game.goals)
#             pygame.display.flip()
#             clock.tick(60)  # Limit to 60 frames per second
#
#             if done:
#                 break
#
#         if episode % target_update == 0:
#             target_net.load_state_dict(policy_net.state_dict())
#
#         # Save the model periodically (optional)
#         #if episode % 50 == 0:
#             #torch.save(policy_net.state_dict(), f"policy_net_{episode}.pth")
#             #torch.save(target_net.state_dict(), f"target_net_{episode}.pth")
#
#     # Save the final model
#     torch.save(policy_net.state_dict(), "policy_net.pth")
#     torch.save(target_net.state_dict(), "target_net.pth")
#
#     print("Training completed")


def train_dqn_bot_vs_bot(state_dim, n_actions, load_model=False):
    # Initialize the environment
    pygame.init()
    board_image = pygame.image.load("assets/board.png")
    board_width, board_height = board_image.get_rect().size
    screen = pygame.display.set_mode((board_width, board_height))
    pygame.display.set_caption("Air Hockey Training - Bot vs Bot")

    gui = GUICore(screen, board_image, board_width, board_height)
    game = GameCore(gui, board_width, board_height)

    # Define hyperparameters
    n_episodes = 5000
    gamma = 0.99
    epsilon_min = 0.01
    epsilon = 1
    epsilon_decay = 0.995
    min_episodes = 5
    update_step = 5
    update_repeats = 30
    memory_capacity = 50000
    batch_size = 128
    lr_step = 100
    lr_gamma = 0.9
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks and memory for both bots
    policy_net1 = DQN(state_dim, n_actions).to(device)
    target_net1 = DQN(state_dim, n_actions).to(device)
    policy_net2 = DQN(state_dim, n_actions).to(device)
    target_net2 = DQN(state_dim, n_actions).to(device)

    for param in target_net1.parameters():
        param.requires_grad = False
    for param in target_net2.parameters():
        param.requires_grad = False

    if load_model:
        try:
            policy_net1.load_state_dict(torch.load("policy_net1_100.pth"))
            target_net1.load_state_dict(torch.load("target_net1_100.pth"))
            policy_net2.load_state_dict(torch.load("policy_net2_100.pth"))
            target_net2.load_state_dict(torch.load("target_net2_100.pth"))
            print("Loaded model from saved files.")
        except FileNotFoundError:
            print("Saved model not found. Starting training from scratch.")

    target_net1.eval()
    target_net2.eval()

    optimizer1 = optim.Adam(policy_net1.parameters(), lr=lr)
    optimizer2 = optim.Adam(policy_net2.parameters(), lr=lr)

    scheduler1 = StepLR(optimizer1, step_size=lr_step, gamma=lr_gamma)
    scheduler2 = StepLR(optimizer2, step_size=lr_step, gamma=lr_gamma)

    memory1 = ReplayBuffer(memory_capacity)
    memory2 = ReplayBuffer(memory_capacity)

    steps_done = 0
    clock = pygame.time.Clock()

    for episode in range(n_episodes):
        print("Episode:" + str(episode))
        game.reset_game()
        state = game.get_state()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action1 = select_action(state, target_net1, epsilon, n_actions)
            action2 = select_action(state, target_net2, epsilon, n_actions)
            steps_done += 1

            game.take_action(action1.item(), 1)  # Bot 1 controls paddle 1
            game.take_action(action2.item(), 2)  # Bot 2 controls paddle 2

            game.update_game_state()

            next_state = game.get_state()
            next_state = (
                torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            )

            reward1 = game.get_reward(1)
            reward2 = game.get_reward(2)

            if reward1 > 0.5 or reward2 > 0.5:
                print("+++++++++++++++++")
                print("Episode:" + str(episode) + " Steps:" + str(steps_done))
                print("Reward1:" + str(reward1))
                print("Reward2:" + str(reward2))

            current_time = pygame.time.get_ticks()
            if current_time - game.last_hit_time > game.max_no_hit_duration:
                print("Skipping episode due to no puck hit in 20 seconds.")
                reward1 = -1.0  # Penalty for inactivity
                reward2 = -1.0  # Penalty for inactivity
                done = True
            else:
                done = not game.running

            reward1 = torch.tensor([reward1], dtype=torch.float32).to(device)
            reward2 = torch.tensor([reward2], dtype=torch.float32).to(device)

            done = torch.tensor([done], dtype=torch.float32).to(device)

            memory1.push(state, action1, reward1, next_state, done)
            memory2.push(state, action2, reward2, next_state, done)

            gui.update(game.paddle1_pos, game.paddle2_pos, game.puck_pos, game.goals)
            game.draw_q_values(screen, target_net1, target_net2, state, game.action_map)
            pygame.display.flip()
            clock.tick(100)  # Limit to 240 frames per second

            if done:
                break

        if episode >= min_episodes and episode % update_step == 0:
            print("Starting training!")
            for _ in range(update_repeats):
                train(memory1, batch_size, policy_net1, target_net1, optimizer1, gamma)
                train(memory2, batch_size, policy_net2, target_net2, optimizer2, gamma)

            print("Training done!")
            target_net1.load_state_dict(policy_net1.state_dict())
            target_net2.load_state_dict(policy_net2.state_dict())

            scheduler1.step()
            scheduler2.step()

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 50 == 0:
            torch.save(policy_net1.state_dict(), f"policy_net1_{episode}.pth")
            torch.save(target_net1.state_dict(), f"target_net1_{episode}.pth")
            torch.save(policy_net2.state_dict(), f"policy_net2_{episode}.pth")
            torch.save(target_net2.state_dict(), f"target_net2_{episode}.pth")

    # Save the final model
    torch.save(policy_net1.state_dict(), "policy_net1.pth")
    torch.save(target_net1.state_dict(), "target_net1.pth")
    torch.save(policy_net2.state_dict(), "policy_net2.pth")
    torch.save(target_net2.state_dict(), "target_net2.pth")

    print("Training completed")
