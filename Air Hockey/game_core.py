from math import atan2, cos, sin, degrees, radians
from random import randrange

import numpy as np
import pygame


class GameCore:
    def set_random_puck_speed(self):
        self.puck_speed["x"] = randrange(-5, 5, 2)
        self.puck_speed["y"] = randrange(-5, 5)

    def __init__(self, gui, board_width, board_height):
        self.gui = gui
        self.running = True
        self.board_width = board_width
        self.board_height = board_height
        self.paddle_radius = 20
        self.puck_radius = 10
        self.max_speed = 10
        self.paddle1_pos = {"x": 50, "y": board_height // 2}
        self.paddle2_pos = {"x": board_width - 50, "y": board_height // 2}
        self.paddle1_velocity = {"x": 0, "y": 0}
        self.paddle2_velocity = {"x": 0, "y": 0}
        self.puck_pos = {"x": board_width // 2, "y": board_height // 2}
        self.puck_speed = {"x": 0.0, "y": 0.0}
        self.goals = {"left": 0, "right": 0}
        self.max_goals = 5000
        self.goal_size = 200  # Height of the goal area
        self.friction = 0.997  # Friction coefficient to reduce speed over time
        self.set_random_puck_speed()
        self.message = ""

        self.speed = 5  # Speed of the bot
        # Timer for detecting no puck hit
        self.last_hit_time = pygame.time.get_ticks()
        self.max_no_hit_duration = 5000  # 20 seconds


        self.goal_reward = 1.5
        self.puck_distance_reward = 0.05
        self.puck_is_behind = 0.06
        self.collision_with_puck = 0.7
        self.position_on_puck_path = 0.05
        self.puck_standing_still = 0.05

    def update_game_state(self):
        # Apply friction to puck speed
        self.puck_speed["x"] *= self.friction
        self.puck_speed["y"] *= self.friction

        self.puck_pos["x"] += self.puck_speed["x"]
        self.puck_pos["y"] += self.puck_speed["y"]

        # Check for collisions with walls (top and bottom)
        if self.puck_pos["y"] <= self.puck_radius:
            self.puck_pos["y"] = self.puck_radius
            self.puck_speed["y"] = abs(
                self.puck_speed["y"]
            )  # Ensure it's moving downwards
        elif self.puck_pos["y"] >= self.board_height - self.puck_radius:
            self.puck_pos["y"] = self.board_height - self.puck_radius
            self.puck_speed["y"] = -abs(
                self.puck_speed["y"]
            )  # Ensure it's moving upwards

        # Check for collisions with paddles
        if self.check_paddle_collision(
            self.paddle1_pos, self.paddle1_velocity
        ) or self.check_paddle_collision(self.paddle2_pos, self.paddle2_velocity):
            self.last_hit_time = pygame.time.get_ticks()  # Reset the timer on puck hit

        # Check for goal
        if self.puck_pos["x"] <= self.puck_radius:
            if self.is_in_goal_area(self.puck_pos["y"]):
                self.goals["right"] += 1
                self.reset_puck("left")
            else:
                self.puck_pos["x"] = self.puck_radius
                self.puck_speed["x"] = abs(
                    self.puck_speed["x"]
                )  # Ensure it's moving right

        elif self.puck_pos["x"] >= self.board_width - self.puck_radius:
            if self.is_in_goal_area(self.puck_pos["y"]):
                self.goals["left"] += 1
                self.reset_puck("right")
            else:
                self.puck_pos["x"] = self.board_width - self.puck_radius
                self.puck_speed["x"] = -abs(
                    self.puck_speed["x"]
                )  # Ensure it's moving lef

        # Check for game over
        if (
            self.goals["left"] >= self.max_goals
            or self.goals["right"] >= self.max_goals
        ):
            self.running = False

    def check_paddle_collision(self, paddle_pos, paddle_velocity):
        dx = self.puck_pos["x"] - paddle_pos["x"]
        dy = self.puck_pos["y"] - paddle_pos["y"]
        distance = (dx**2 + dy**2) ** 0.5

        if distance <= self.paddle_radius + self.puck_radius:
            angle = atan2(dy, dx)
            speed = (self.puck_speed["x"] ** 2 +
                     self.puck_speed["y"] ** 2) ** 0.5
            # Influence puck speed with paddle velocity
            self.puck_speed["x"] = min(
                self.max_speed, speed * cos(angle) + paddle_velocity["x"]
            )
            self.puck_speed["y"] = min(
                self.max_speed, speed * sin(angle) + paddle_velocity["y"]
            )
            return True
        return False

    def is_in_goal_area(self, y_pos):
        goal_top = self.board_height // 2 - self.goal_size // 2
        goal_bottom = self.board_height // 2 + self.goal_size // 2
        return goal_top < y_pos < goal_bottom

    def reset_puck(self, side):
        self.puck_pos = {"x": self.board_width //
                         2, "y": self.board_height // 2}
        self.puck_speed = {"x": 5.0 if side == "left" else -5.0, "y": 3.0}

    def move_paddle(self, player, x_pos, y_pos):
        if player == 1:
            self.paddle1_velocity = {
                # "x": min(x_pos, self.board_width // 2 - self.paddle_radius)
                # - self.paddle1_pos["x"],
                # "y": min(y_pos, self.board_height // 2 - self.paddle_radius)
                # - self.paddle1_pos["y"],
                "x": x_pos - self.paddle1_pos["x"],
                "y": y_pos - self.paddle1_pos["y"],
            }
            self.paddle1_pos["x"] = max(
                self.paddle_radius,
                min(x_pos, self.board_width // 2 - self.paddle_radius),
            )
            self.paddle1_pos["y"] = max(
                self.paddle_radius, min(
                    y_pos, self.board_height - self.paddle_radius)
            )
        elif player == 2:
            self.paddle2_velocity = {
                "x": x_pos - self.paddle2_pos["x"],
                "y": y_pos - self.paddle2_pos["y"],
            }
            self.paddle2_pos["x"] = max(
                self.board_width // 2 + self.paddle_radius,
                min(x_pos, self.board_width - self.paddle_radius),
            )
            self.paddle2_pos["y"] = max(
                self.paddle_radius, min(
                    y_pos, self.board_height - self.paddle_radius)
            )

    def reset_game(self):
        self.paddle1_pos = {"x": 50, "y": self.board_height // 2}
        self.paddle2_pos = {"x": self.board_width -
                            50, "y": self.board_height // 2}
        self.paddle1_velocity = {"x": 0, "y": 0}
        self.paddle2_velocity = {"x": 0, "y": 0}
        self.puck_pos = {"x": self.board_width //
                         2, "y": self.board_height // 2}
        self.set_random_puck_speed()
        self.last_hit_time = pygame.time.get_ticks()

    def get_state(self, player):
        angle = degrees(atan2(self.puck_speed["y"], self.puck_speed["x"]))
        if player == 1:
            paddlex = self.paddle1_pos["x"]
            paddley = self.paddle1_pos["y"]
        else:
            paddlex = self.paddle2_pos["x"]
            paddley = self.paddle2_pos["y"]

        state = [
            paddlex,
            paddley,
            self.puck_pos["x"],
            self.puck_pos["y"],
            self.puck_speed["x"],
            self.puck_speed["y"],
            angle,
        ]
        return np.array(state, dtype=np.float32)

    def take_action(self, action, player):
        x_move, y_move = action
        x_move = x_move * 5
        y_move = y_move * 5
        if player == 1:
            self.move_paddle(
                1, self.paddle1_pos["x"] +
                x_move, self.paddle1_pos["y"] + y_move
            )
        else:
            self.move_paddle(
                2, self.paddle2_pos["x"] +
                x_move, self.paddle2_pos["y"] + y_move
            )
    def get_reward(self, player):
        reward = 0.0
        predicted_path = self.predict_puck_path(20)

        if player == 1:  # Left bot
            # Left bot scores a goal
            if self.puck_pos["x"] >= self.board_width - self.puck_radius:
                if self.is_in_goal_area(self.puck_pos["y"]):
                    reward += self.goal_reward
                    self.print_message("GOAAAAAAL by Left Bot")

            # Right bot scores a goal
            elif self.puck_pos["x"] <= self.puck_radius:
                if self.is_in_goal_area(self.puck_pos["y"]):
                    reward -= self.goal_reward 
                    self.print_message("Right Bot scored a goal")

            # When puck in left bot's half court, decrease distance to puck
            if self.puck_pos["x"] < self.board_width / 2:
                distance = np.sqrt(
                    (self.paddle1_pos["x"] - self.puck_pos["x"]) ** 2
                    + (self.paddle1_pos["y"] - self.puck_pos["y"]) ** 2
                )
                reward += self.puck_distance_reward * (1 - distance / self.board_width)

            # Puck is behind the left bot
            if self.puck_pos["x"] < self.paddle1_pos["x"]:
                reward -= self.puck_is_behind 
                self.print_message("Puck is behind the Left Bot")

            # Detect a collision between left bot and puck
            distance_paddle_puck = np.sqrt(
                (self.paddle1_pos["x"] - self.puck_pos["x"]) ** 2
                + (self.paddle1_pos["y"] - self.puck_pos["y"]) ** 2
            )
            if distance_paddle_puck <= self.paddle_radius + self.puck_radius:
                reward += self.collision_with_puck
                self.print_message("Left Bot hits the puck")

            # Reward for positioning near the predicted path of the puck
            for position in predicted_path:
                distance_to_predicted = np.sqrt(
                    (self.paddle1_pos["x"] - position["x"]) ** 2
                    + (self.paddle1_pos["y"] - position["y"]) ** 2
                )
                reward += self.position_on_puck_path * (1 - distance_to_predicted / (self.board_width/2))

            # Punish puck standing still
            puck_speed = np.sqrt(
                self.puck_speed["x"] ** 2 + self.puck_speed["y"] ** 2)
            if puck_speed < 0.1:
                reward -= self.puck_standing_still
                self.print_message("Puck is standing still")

        elif player == 2:  # Right bot
            # Right bot scores a goal
            if self.puck_pos["x"] <= self.puck_radius:
                if self.is_in_goal_area(self.puck_pos["y"]):
                    reward += self.goal_reward
                    self.print_message("GOAAAAAAL by Right Bot")

            # Left bot scores a goal
            elif self.puck_pos["x"] >= self.board_width - self.puck_radius:
                if self.is_in_goal_area(self.puck_pos["y"]):
                    reward -= self.goal_reward
                    self.print_message("Left Bot scored a goal")

            # When puck in right bot's half court, decrease distance to puck
            if self.puck_pos["x"] > self.board_width / 2:
                distance = np.sqrt(
                    (self.paddle2_pos["x"] - self.puck_pos["x"]) ** 2
                    + (self.paddle2_pos["y"] - self.puck_pos["y"]) ** 2
                )
                reward += self.puck_distance_reward * (1 - distance / (self.board_width/2))

            # Puck is behind the right bot
            if self.puck_pos["x"] > self.paddle2_pos["x"]:
                reward -= self.puck_distance_reward
                self.print_message("Puck is behind the Right Bot")

            # Detect a collision between right bot and puck
            distance_paddle_puck = np.sqrt(
                (self.paddle2_pos["x"] - self.puck_pos["x"]) ** 2
                + (self.paddle2_pos["y"] - self.puck_pos["y"]) ** 2
            )
            if distance_paddle_puck <= self.paddle_radius + self.puck_radius:
                reward += self.collision_with_puck 
                self.print_message("Right Bot hits the puck")

            # Reward for positioning near the predicted path of the puck
            for position in predicted_path:
                distance_to_predicted = np.sqrt(
                    (self.paddle2_pos["x"] - position["x"]) ** 2
                    + (self.paddle2_pos["y"] - position["y"]) ** 2
                )
                reward += self.position_on_puck_path * (1 - distance_to_predicted / self.board_width)

            # Punish puck standing still
            puck_speed = np.sqrt(
                self.puck_speed["x"] ** 2 + self.puck_speed["y"] ** 2)
            if puck_speed < 0.1:
                reward -= self.puck_standing_still
                self.print_message("Puck is standing still")

        return reward

    def print_message(self, message):
        if message != self.message:
            self.message = message
            print(message)

    def predict_puck_path(self, time_steps):
        predicted_path = []
        x, y = self.puck_pos["x"], self.puck_pos["y"]
        vx, vy = self.puck_speed["x"], self.puck_speed["y"]

        for _ in range(time_steps):
            x += vx
            y += vy

            # Handle wall collisions
            if y <= self.puck_radius or y >= self.board_height - self.puck_radius:
                vy = -vy
            if x <= self.puck_radius or x >= self.board_width - self.puck_radius:
                vx = -vx

            predicted_path.append({"x": x, "y": y})

        return predicted_path

    def draw_q_values(self, screen, policy_net1, policy_net2, state, action_map):
        font = pygame.font.Font(None, 24)
        for action in range(len(action_map)):
            q_value1 = policy_net1(state).detach().cpu().numpy()[0, action]
            q_value2 = policy_net2(state).detach().cpu().numpy()[0, action]
            action_name = action_map[action]

            q_value_text1 = font.render(
                f"P1 {action_name}: {q_value1:.2f}", True, (0, 0, 0)
            )
            q_value_text2 = font.render(
                f"P2 {action_name}: {q_value2:.2f}", True, (0, 0, 0)
            )

            screen.blit(q_value_text1, (10, 10 + action * 25))
            screen.blit(q_value_text2, (10, 10 +
                        len(action_map) * 25 + action * 25))
