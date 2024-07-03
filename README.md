# Air Hockey AI (Work in Progress)

This repository is dedicated to the development of AI agents for playing the game of Air Hockey. The goal of this project is to implement intelligent agents capable of playing Air Hockey.

## Project Overview

Air Hockey is a fast-paced game that requires quick reactions, precision, and strategic thinking. The project aims to develop AI agents capable of playing Air Hockey against either another AI agent or a human player. The agents will learn to react to the dynamics of the game, anticipate opponent movements, and make strategic decisions to score goals while defending their own goal.

## Features (Planned)

- Implementation of the Air Hockey game using PyGame library.
- Training of AI agents using Q-Learning and NEAT algorithms.
- Development of reward and penalty systems to motivate AI agents to learn optimal behavior.
- Evaluation of AI agent performance in real-time gameplay scenarios.

## Technologies and Tools

- **Programming Language**: Python
- **Libraries and Tools**:
  - PyTorch for deep learning
  - Pygame for the graphical interface and game simulation
  - Sklearn for data processing

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- `pip` (Python package installer)

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/air-hockey-ai.git
   cd air-hockey-ai


## Contributing

Contributions to the project are welcome! If you have any ideas for improvements or would like to report issues, please open a GitHub issue or submit a pull request.

## Authors

- Teodor Vidaković - [GitHub](https://github.com/Tekisha)
- Danilo Cvijetić - [GitHub](https://github.com/c-danil0o)

## Video Demonstration

![Game Demo](readme_assets/air_hockey.gif)

## Documentation

### Introduction

#### Project Description

- **Game Development**: Developed the Air Hockey game using Pygame to simulate the game environment. This simulation provides a realistic representation of dynamics, including physical interactions between the puck, paddles, and walls.
- **AI Development**: Worked on developing and training AI agents for game simulation using deep learning techniques. Using neural networks, the agents are trained to recognize patterns and optimize their behavior during the game.
- **MADDPG**: The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) method is used to coordinate multiple agents in the game environment. This algorithm allows agents to act in a shared space and communicate and coordinate their actions for better results.

#### Project Goals

- **Strategy Development**: Develop effective strategies for controlling two agents in the simulation. The goal is for the agents to be able to make quick and precise decisions during the game, optimizing their movements and strategies for scoring goals and defending their goal.
- **Rewards and Penalties**: Implement rewards and penalties to optimize agent behavior. Different scenarios in the game, such as scoring a goal, holding the puck, and colliding with the puck, are associated with specific rewards and penalties that help agents learn more effective tactics and behaviors.

#### Motivation

- **AI in Dynamic Environments**: Developing intelligent agents for games provides insights into the application of artificial intelligence in dynamic and unpredictable environments. This research can contribute to advancements in various fields, including robotics, autonomous vehicles, and other systems requiring quick decision-making.
- **Games as Testing Grounds**: Games like Air Hockey offer an ideal environment for testing and developing algorithms for multi-agent systems, as they involve complex interactions and strategies that can be applied to a broader range of real-world applications.

#### Challenges

- **Balancing Rewards and Penalties**: One of the main challenges is balancing rewards and penalties to achieve optimal agent behavior. Excessive penalties can demotivate the agent, while too many rewards can lead to suboptimal strategies.
- **Adaptability**: Ensuring that agents can quickly adapt their strategies in the changing game environment requires sophisticated learning and adaptation algorithms.
- **Training Resources**: Training AI agents requires significant computational resources, including powerful GPUs and ample time for simulations and learning. Resource constraints can slow down the training process and make it challenging to achieve optimal results.

### Learning Algorithm and Logic

#### DDPG (Deep Deterministic Policy Gradient)

- **Algorithm for Continuous Actions**:
  - DDPG is an off-policy reinforcement learning algorithm used for problems with continuous actions.
  - Combines ideas from DQN (Deep Q-Network) and DPG (Deterministic Policy Gradient) algorithms for efficient learning in large action spaces.

- **Actor and Critic Networks**:
  - **Actor**: Neural network mapping states to actions. The actor generates actions for the agent to execute.
  - **Critic**: Neural network evaluating the Q-function, i.e., the expected sum of future rewards for a given state-action pair. The critic assesses the quality of a given action in a given state.
  - The algorithm uses two sets of actor and critic networks: main networks (updated during training) and target networks (updated more slowly to stabilize learning).
  - Actor loss is computed as the negative assessment of the critic's Q-value, while critic loss is computed using the TD (Temporal Difference) error.

#### MADDPG (Multi-Agent DDPG)

- **Extension for Multi-Agent Environments**:
  - MADDPG extends the DDPG algorithm to multi-agent environments.
  - Each agent has its actor network, while critic networks can be shared or specific to each agent.

- **Action Coordination and Information Sharing**:
  - Agents share information about their states and actions to improve the learning process. This allows each agent to consider the actions of other agents when making its own decisions.
  - The critic uses information about the actions of all agents to evaluate the Q-value, enabling coordination among agents.
  - Information sharing reduces uncertainty and improves collective system performance.

- **Network Updates and Learning Stability**:
  - As in DDPG, target networks are used to stabilize learning. These networks are updated using a soft update technique, where target network parameters slowly approach those of the main network.
  - Experience replay is used, where agents store their past experiences and use them to train the networks. This allows for more efficient and stable learning by reducing the correlation between successive experiences.

### Rewards and Penalties Implementation

#### Basic Rewards

- **Goal Reward**: Awarded to the agent when it scores a goal. This reward encourages the agent to achieve objectives and play effectively.
- **Proximity to Puck Reward**: Positive reward when the agent gets closer to the puck. This reward helps agents actively participate in the game and maintain control over the puck.
- **Puck Collision Reward**: Positive reward when the agent collides with the puck, encouraging it to actively track and hit the puck.
- **Directing Puck Towards Opponent Goal Reward**: Awarded when the agent successfully directs the puck towards the opponent's goal during the simulated puck path.
- **Puck Acceleration Reward**: Positive reward when the agent increases the puck's speed.

#### Penalties

- **Dribbling in Own Half Penalty**: Points deducted for holding the puck in its own half for too long. This penalty encourages the agent to move towards the opponent's goal.
- **Slow Puck Approach Penalty**: Penalty if the agent approaches the puck too slowly, encouraging faster and more efficient movement.
- **Puck Standing Still Penalty**: Penalty when the puck stands still, encouraging continuous activity and game dynamics.
- **Misleading Puck Direction Penalty**: Penalty for directing the puck towards its own goal, preventing own goals.
- **Puck Behind Agent Penalty**: Penalty when the puck is behind the agent, encouraging the agent to position itself in front of the puck.
- **Slowing Puck Penalty**: Penalty when the agent decreases the puck's speed.

### Code Structure

- **Class `GameCore`**:
  - Manages the game state, including puck position and speed, paddle positions, collision detection, and reward calculation.
  - Key methods: `update_game_state`, `get_reward`, `move_paddle`, `check_paddle_collision`.

- **Classes `ActorModel` and `CriticModel`**:
  - Defines the neural networks for the agent.
  - Key methods: `forward`, `reset_parameters`.

- **Class `Agent_DDPG`**:
  - Manages agent learning, including actions and model updates.
  - Key methods: `act`, `learn`, `soft_update`.

- **Class `Agent_MADDPG`**:
  - Coordinates multiple agents and their collective learning.
  - Key methods: `step`, `act`, `learn`.

- **Class `GUICore`**:
  - Provides the graphical interface for the game, including displaying game state, paddles, puck, and scores.
  - Key methods: `update`, `close`, `draw_predicted_path`.

- **Class `ReplayBuffer`**:
  - Implements a repository for storing agent experiences (states, actions, rewards, next states, episode end flags).
  - Key methods: `add`, `sample`.

- **Class `OUNoise`**:
  - Implements Ornstein-Uhlenbeck noise for exploration during agent training.
  - Key methods: `reset`, `sample`.

### Results and Discussion

#### Agent Performance

- During training, agents showed solid performance in learning the basic rules of the game. Their ability to control paddles, collide with the puck, and score goals significantly improved.
- Agents developed effective strategies during training, including better movement coordination, quicker responses to the puck's position, and optimized shots towards the opponent's goal.
- Although agent performance is satisfactory, further training could enhance their strategy and efficiency.

#### Challenges and Issues

- **Hyperparameter Tuning**: Adjusting hyperparameters was a key challenge. Each of these parameters significantly impacted the algorithm's stability and convergence speed.
- **Learning Stability**: Deep learning algorithms, especially those used in multi-agent environments, are prone to stability issues. Agent behavior often oscillated, particularly in the early training stages. Stability improved through techniques like experience replay.
- **Exploration Issues**: Agents occasionally exploited learned strategies too much, leading to suboptimal results. Implementing Ornstein-Uhlenbeck noise helped overcome this, allowing agents to explore a broader range of possible actions.
- **Resources**: Training deep networks is computationally intensive. Limited resources, including hardware capabilities and time, posed a significant challenge. We used NVIDIA CUDA on an RTX 3050 Ti GPU to accelerate training and reduce system load.

### Conclusion

#### Achievements

- Successfully implemented AI agents capable of playing Air Hockey efficiently. Agents demonstrated the ability to control paddles, hit the puck, score goals, and develop strategies for optimal play.
- The application of deep learning techniques, specifically DDPG and MADDPG algorithms, enabled agents to learn from experiences and continuously improve their performance. This project demonstrated the effectiveness of these techniques in complex multi-agent environments.
- Agents successfully integrated various rewards and penalties to optimize their behavior, showing progress in developing sophisticated strategies through training.
- Utilized hardware resources, including NVIDIA CUDA on an RTX 3050 Ti GPU, to accelerate training and achieve better performance, enabling faster model convergence and more efficient resource use.

#### Future Work

- Further algorithm improvements can enhance agent performance. We plan to experiment with different hyperparameters.
- We plan to integrate more complex scenarios and environments to prepare agents for potential real-world applications in robotics and automation.
- We will explore the use of other libraries and tools to improve the graphical interface and simulation, enhancing visualization and interactivity during training.
- We plan to explore distributed training on multiple GPUs or using cloud resources to speed up the training process and work with larger models and more complex environments.
- Creating bots with different strategies to train against each other, providing more diverse training and developing advanced tactical capabilities in agents.
