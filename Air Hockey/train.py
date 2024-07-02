import pygame
import numpy as np

from agent import Agent_MADDPG
from game_core import GameCore
from gui_core import GUICore


def train_maddpg(state_dim, n_actions, load_model=True, show_gui_after_episodes=0, save_interval=100):
    # Initialize the environment
    pygame.init()
    board_image = pygame.image.load("assets/board.png")
    board_width, board_height = board_image.get_rect().size
    screen = pygame.display.set_mode((board_width, board_height))
    pygame.display.set_caption("Air Hockey Training - Bot vs Bot")

    gui = GUICore(screen, board_image, board_width, board_height)
    game = GameCore(gui, board_width, board_height)

    # Define hyperparameters
    n_episodes = 15000
    max_episode_duration = 30 * 1000  # 40 seconds in milliseconds

    agent = Agent_MADDPG(n_actions, state_dim)

    if load_model:
        agent.load_agents()

    clock = pygame.time.Clock()
    start_epoch = agent.epoch

    try:
        for episode in range(start_epoch, n_episodes):
            agent.epoch = episode
            print("Episode:" + str(episode))
            steps_done = 0
            game.reset_game()
            state_1 = game.get_state(1)
            state_2 = game.get_state(2)
            all_states = np.stack([state_1, state_2], axis=0)
            done = False
            # episode_start_time = pygame.time.get_ticks()

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save_agents()
                        pygame.quit()
                        return

                action = agent.act(all_states)
                steps_done += 1
                action1 = (action[0][0].item(), action[0][1].item())
                action2 = (action[0][2].item(), action[0][3].item())

                game.take_action(action1, 1)  # Bot 1 controls paddle 1
                game.take_action(action2, 2)  # Bot 2 controls paddle 2

                game.update_game_state()

                all_next_state = np.stack(
                    [game.get_state(1), game.get_state(2)], axis=0)

                reward1 = game.get_reward(1)
                reward2 = game.get_reward(2)

                # if reward1 > 0.9 or reward2 > 0.9:
                #     print("+++++++++++++++++")
                #     print("Episode:" + str(episode) + " Steps:" + str(steps_done))
                #     print("Reward1:" + str(reward1))
                #     print("Reward2:" + str(reward2))

                current_time = pygame.time.get_ticks()
                if current_time - game.last_hit_time > game.max_no_hit_duration:
                    print(
                        f"Skipping episode due to no puck hit in {game.max_no_hit_duration / 1000} seconds.")
                    reward1 = -0.6  # Penalty for inactivity
                    reward2 = -0.6  # Penalty for inactivity
                    done = True
                # elif current_time - episode_start_time > max_episode_duration:
                #     print("Ending episode due to max duration reached.")
                #     done = True
                else:
                    done = not game.running

                agent.step(all_states, action, [
                           reward1, reward2], all_next_state, [done, done])
                all_states = all_next_state

                if episode >= show_gui_after_episodes:
                    gui.update(game.paddle1_pos, game.paddle2_pos,
                               game.puck_pos, game.goals, game.predicted_path)
                    pygame.display.flip()

                if steps_done >= 5000:
                    done = True
                # gui.update(game.paddle1_pos, game.paddle2_pos,
                    # game.puck_pos, game.goals)
                # game.draw_q_values(screen, target_net1,
                #                    target_net2, state, game.action_map)
                # pygame.display.flip()
                clock.tick(500)  # Limit to 240 frames per second

                if done:
                    break

            if episode % save_interval == 0:
                agent.save_agents()
    except KeyboardInterrupt:
        print("Training interrupted, saving agents...")
        agent.save_agents()

    print("Training completed")
    pygame.quit()
