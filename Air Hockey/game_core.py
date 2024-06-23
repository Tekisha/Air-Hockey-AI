from math import atan2, cos, sin
from random import randrange


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
        self.paddle1_pos = {"x": 50, "y": board_height // 2}
        self.paddle2_pos = {"x": board_width - 50, "y": board_height // 2}
        self.paddle1_velocity = {"x": 0, "y": 0}
        self.paddle2_velocity = {"x": 0, "y": 0}
        self.puck_pos = {"x": board_width // 2, "y": board_height // 2}
        self.puck_speed = {"x": 0.0, "y": 0.0}
        self.goals = {"left": 0, "right": 0}
        self.max_goals = 7
        self.goal_size = 200  # Height of the goal area
        self.friction = 0.997  # Friction coefficient to reduce speed over time
        self.set_random_puck_speed()

    def update_game_state(self):
        # Apply friction to puck speed
        self.puck_speed["x"] *= self.friction
        self.puck_speed["y"] *= self.friction

        self.puck_pos["x"] += self.puck_speed["x"]
        self.puck_pos["y"] += self.puck_speed["y"]

        # Check for collisions with walls (top and bottom)
        if (
            self.puck_pos["y"] <= self.puck_radius
            or self.puck_pos["y"] >= self.board_height - self.puck_radius
        ):
            self.puck_speed["y"] = -self.puck_speed["y"]

        # Check for collisions with paddles
        self.check_paddle_collision(self.paddle1_pos, self.paddle1_velocity)
        self.check_paddle_collision(self.paddle2_pos, self.paddle2_velocity)

        # Check for goal
        if self.puck_pos["x"] <= self.puck_radius:
            if self.is_in_goal_area(self.puck_pos["y"]):
                self.goals["right"] += 1
                self.reset_puck("left")
            else:
                self.puck_speed["x"] = -self.puck_speed["x"]

        elif self.puck_pos["x"] >= self.board_width - self.puck_radius:
            if self.is_in_goal_area(self.puck_pos["y"]):
                self.goals["left"] += 1
                self.reset_puck("right")
            else:
                self.puck_speed["x"] = -self.puck_speed["x"]

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
            speed = (self.puck_speed["x"] ** 2 + self.puck_speed["y"] ** 2) ** 0.5
            # Influence puck speed with paddle velocity
            self.puck_speed["x"] = speed * cos(angle) + paddle_velocity["x"]
            self.puck_speed["y"] = speed * sin(angle) + paddle_velocity["y"]

    def is_in_goal_area(self, y_pos):
        goal_top = self.board_height // 2 - self.goal_size // 2
        goal_bottom = self.board_height // 2 + self.goal_size // 2
        return goal_top < y_pos < goal_bottom

    def reset_puck(self, side):
        self.puck_pos = {"x": self.board_width // 2, "y": self.board_height // 2}
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
                self.paddle_radius, min(y_pos, self.board_height - self.paddle_radius)
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
                self.paddle_radius, min(y_pos, self.board_height - self.paddle_radius)
            )

    def reset_game(self):
        self.paddle1_pos = {"x": 50, "y": self.board_height // 2}
        self.paddle2_pos = {"x": self.board_width - 50, "y": self.board_height // 2}
        self.paddle1_velocity = {"x": 0, "y": 0}
        self.paddle2_velocity = {"x": 0, "y": 0}
        self.puck_pos = {"x": self.board_width // 2, "y": self.board_height // 2}
        self.set_random_puck_speed()
