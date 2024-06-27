import pygame
import torch
from torch import optim

from dqn import DQN, ReplayBuffer, select_action, optimize_model
from game_core import GameCore
from gui_core import GUICore


def train_dqn(state_dim, n_actions):
    # Initialize the environment
    pygame.init()
    board_image = pygame.image.load("assets/board.png")
    board_width, board_height = board_image.get_rect().size
    screen = pygame.display.set_mode((board_width, board_height))
    pygame.display.set_caption("Air Hockey Training")

    gui = GUICore(screen, board_image, board_width, board_height)
    game = GameCore(gui, board_width, board_height)

    # Define hyperparameters
    n_episodes = 500
    max_steps = 10000
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 500
    target_update = 10
    memory_capacity = 10000
    batch_size = 128

    # Initialize networks and memory
    policy_net = DQN(state_dim, n_actions)
    target_net = DQN(state_dim, n_actions)
    policy_net.load_state_dict(torch.load("policy_net.pth"))
    target_net.load_state_dict(torch.load("target_net.pth"))
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayBuffer(memory_capacity)

    steps_done = 0
    clock = pygame.time.Clock()

    for episode in range(n_episodes):
        game.reset_game()
        state = game.get_state()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Correct tensor conversion

        for t in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    game.move_paddle(1, mouse_x, mouse_y)  # Player controls paddle 1

            action = select_action(state, policy_net, steps_done, epsilon_end, epsilon_start, epsilon_decay, n_actions)
            steps_done += 1
            game.take_action(action.item(), 2)  # AI controls paddle 2

            game.update_game_state()

            next_state = game.get_state()
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            reward = game.get_reward()
            reward = torch.tensor([reward], dtype=torch.float32)

            done = not game.running
            done = torch.tensor([done], dtype=torch.float32)

            memory.push(state, action, reward, next_state, done)

            state = next_state

            optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma)

            gui.update(game.paddle1_pos, game.paddle2_pos, game.puck_pos, game.goals)
            pygame.display.flip()
            clock.tick(60)  # Limit to 60 frames per second

            if done:
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save the model periodically (optional)
        #if episode % 50 == 0:
            #torch.save(policy_net.state_dict(), f"policy_net_{episode}.pth")
            #torch.save(target_net.state_dict(), f"target_net_{episode}.pth")

    # Save the final model
    torch.save(policy_net.state_dict(), "policy_net.pth")
    torch.save(target_net.state_dict(), "target_net.pth")

    print("Training completed")

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
    max_steps = 50000
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 5000
    target_update = 10
    memory_capacity = 10000
    batch_size = 128

    # Initialize networks and memory for both bots
    policy_net1 = DQN(state_dim, n_actions)
    target_net1 = DQN(state_dim, n_actions)
    policy_net2 = DQN(state_dim, n_actions)
    target_net2 = DQN(state_dim, n_actions)

    if load_model:
        try:
            policy_net1.load_state_dict(torch.load("policy_net1.pth"))
            target_net1.load_state_dict(torch.load("target_net1.pth"))
            policy_net2.load_state_dict(torch.load("policy_net2.pth"))
            target_net2.load_state_dict(torch.load("target_net2.pth"))
            print("Loaded model from saved files.")
        except FileNotFoundError:
            print("Saved model not found. Starting training from scratch.")

    target_net1.eval()
    target_net2.eval()

    optimizer1 = optim.Adam(policy_net1.parameters())
    optimizer2 = optim.Adam(policy_net2.parameters())
    memory1 = ReplayBuffer(memory_capacity)
    memory2 = ReplayBuffer(memory_capacity)

    steps_done1 = 0
    steps_done2 = 0
    clock = pygame.time.Clock()

    for episode in range(n_episodes):
        print("Episode:"+str(episode))
        game.reset_game()
        state = game.get_state()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        for t in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action1 = select_action(state, policy_net1, steps_done1, epsilon_end, epsilon_start, epsilon_decay, n_actions)
            action2 = select_action(state, policy_net2, steps_done2, epsilon_end, epsilon_start, epsilon_decay, n_actions)
            steps_done1 += 1
            steps_done2 += 1

            game.take_action(action1.item(), 1)  # Bot 1 controls paddle 1
            game.take_action(action2.item(), 2)  # Bot 2 controls paddle 2

            game.update_game_state()

            next_state = game.get_state()
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            reward1 = game.get_reward(1)
            reward2 = game.get_reward(2)

            print("+++++++++++++++++")
            print("Episode:" + str(episode) + "Steps1:" + str(steps_done1) + "Steps2" + str(steps_done2))
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

            reward1 = torch.tensor([reward1], dtype=torch.float32)
            reward2 = torch.tensor([reward2], dtype=torch.float32)

            done = torch.tensor([done], dtype=torch.float32)

            memory1.push(state, action1, reward1, next_state, done)
            memory2.push(state, action2, reward2, next_state, done)

            state = next_state

            optimize_model(memory1, batch_size, policy_net1, target_net1, optimizer1, gamma)
            optimize_model(memory2, batch_size, policy_net2, target_net2, optimizer2, gamma)

            gui.update(game.paddle1_pos, game.paddle2_pos, game.puck_pos, game.goals)
            pygame.display.flip()
            clock.tick(60)  # Limit to 60 frames per second

            if done:
                break

        if episode % target_update == 0:
            target_net1.load_state_dict(policy_net1.state_dict())
            target_net2.load_state_dict(policy_net2.state_dict())

        # Save the model periodically
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




