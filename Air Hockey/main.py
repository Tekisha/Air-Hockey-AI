import sys

from test_model import test_model
from train import train_maddpg


# Main function
def main():
    global state_dim, n_actions

    state_dim = 11
    n_actions = 2

    if len(sys.argv) < 2:
        print("Usage: python main.py [train|play]")
        return
    mode = sys.argv[1]
    # mode = ""
    if mode == "train":
        train_maddpg(state_dim, n_actions)
    elif mode == "play":
        test_model(state_dim, n_actions)
    else:
        print("Unknown mode:", mode)
        print("Usage: python main.py [train|play]")


if __name__ == "__main__":
    main()
