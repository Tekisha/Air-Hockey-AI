import sys

import pygame
import torch

from dqn import DQN
from game_core import GameCore
from gui_core import GUICore
from test_model import test_model
from train_dqn import train_dqn_bot_vs_bot


# Main function
def main():
    global state_dim, n_actions

    state_dim = 9
    n_actions = 9

    # if len(sys.argv) < 2:
    #     print("Usage: python main.py [train|test_player_vs_bot|test_bot_vs_bot]")
    #     return
    train_dqn_bot_vs_bot(state_dim, n_actions)
    # mode = sys.argv[1]
    mode = ""
    if mode == "train":
        train_dqn_bot_vs_bot(state_dim, n_actions)
    elif mode == "test_player_vs_bot":
        policy_net = DQN(state_dim, n_actions)
        policy_net.load_state_dict(torch.load("policy_net.pth"))
        policy_net.eval()
        test_model(policy_net, mode='player_vs_bot')
    elif mode == "test_bot_vs_bot":
        policy_net = DQN(state_dim, n_actions)
        policy_net.load_state_dict(torch.load("policy_net.pth"))
        policy_net.eval()
        test_model(policy_net, mode='bot_vs_bot')
    else:
        print("Unknown mode:", mode)
        print("Usage: python main.py [train|test_player_vs_bot|test_bot_vs_bot]")


if __name__ == "__main__":
    main()
