import math
import random
import numpy as np
from Structure_project.drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv, MDPEnv
from tkinter import *
from tkinter.messagebox import showinfo
import warnings


class TicTacToe(SingleAgentEnv):
    def __init__(self, nb_cells: int = 3 ** 9):
        self.nb_cells = nb_cells
        self.current_cell = 0
        self.step_count = 0
        self.win_rate = 0.0
        self.game_played = 0

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        if self.step_count >= 9:
            return True
        values = self.values_cases()
        for i in range(3):
            if (values[i] == values[i + 3] and values[i + 3] == values[i + (2 * 3)] and (
                    values[i] == 1 or values[i] == 2)) or \
                    (values[i * 3] == values[(i * 3) + 1] and values[(i * 3) + 1] == values[(i * 3) + 2] and (
                            values[i * 3] == 1 or values[i * 3] == 2)) or \
                    (values[i] == values[4] and values[4] == values[8 - i] and (values[i] == 1 or values[i] == 2)):
                return True
        return False

    def available_actions_ids(self) -> np.ndarray:
        aa = []
        values = self.values_cases()
        for i in range(9):
            if values[i] == 0:
                aa.append(i)
        return np.array(aa)

    def act_with_action_id(self, action_id: int):
        self.current_cell += 3 ** (8 - action_id)
        self.step_count += 1
        if not self.is_game_over():
            aa = self.available_actions_ids()
            random_move = random.randint(0, len(aa) - 1)
            self.current_cell += 2 * (3 ** (8 - aa[random_move]))
            self.step_count += 1

    def score(self) -> float:
        values = self.values_cases()
        for i in range(3):
            if (values[i] == values[i + 3] and values[i + 3] == values[i + (2 * 3)] and values[i] == 1) or \
                    (values[i * 3] == values[(i * 3) + 1] and values[(i * 3) + 1] == values[(i * 3) + 2] and values[
                        i * 3] == 1) or \
                    (values[i] == values[4] and values[4] == values[8 - i] and values[i] == 1):
                return 1.0
            elif (values[i] == values[i + 3] and values[i + 3] == values[i + (2 * 3)] and values[i] == 2) or \
                    (values[i * 3] == values[(i * 3) + 1] and values[(i * 3) + 1] == values[(i * 3) + 2] and values[
                        i * 3] == 2) or \
                    (values[i] == values[4] and values[4] == values[8 - i] and values[i] == 2):
                return -1.0
        return 0.0

    def reset(self):
        self.current_cell = 0
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        values = self.values_cases()
        for i in range(3):
            for j in range(3):
                if values[(i * 3) + j] == 0:
                    print("_", end="")
                elif values[(i * 3) + j] == 1:
                    print("X", end="")
                else:
                    print("O", end="")
            print()
        if self.is_game_over():
            self.win_rate += self.score()
            self.game_played += 1
            print(f'win rate : {self.win_rate / self.game_played if self.game_played > 0 else 0}')
            print(f'game played : {self.game_played}')

    def reset_random(self):
        while True:
            self.reset()
            aa = self.available_actions_ids()
            while len(aa) > 0:
                random_move = random.randint(0, len(aa) - 1)
                rand = random.randint(0, 1)
                if self.step_count % 2 == 0:
                    self.current_cell += rand * 3 ** (8 - aa[random_move])
                else:
                    self.current_cell += rand * 2 * (3 ** (8 - aa[random_move]))
                self.step_count += rand
                aa = np.delete(aa, random_move)
            if not self.is_game_over():
                break

    def values_cases(self) -> np.ndarray:
        values = np.zeros(9)
        tmp_state = self.current_cell
        for i in range(9):
            values[i] = tmp_state // 3 ** (8 - i)
            tmp_state %= 3 ** (8 - i)
        return values

    def init_tkinter(self, pi):
        root = Tk()
        root.title("Tic-Tac-Toe")

        label1 = Label(root, text="X: Le joueur", font=("Courrier", 15))
        label1.grid(row=0, column=0)
        label2 = Label(root, text="0: L'IA", font=("Courrier", 15))
        label2.grid(row=0, column=1)
        b1 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(0, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b1.grid(row=1, column=0)
        b2 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(1, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b2.grid(row=1, column=1)
        b3 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(2, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b3.grid(row=1, column=2)
        b4 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(3, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b4.grid(row=2, column=0)
        b5 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(4, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b5.grid(row=2, column=1)
        b6 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(5, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b6.grid(row=2, column=2)
        b7 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(6, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b7.grid(row=3, column=0)
        b8 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(7, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b8.grid(row=3, column=1)
        b9 = Button(root, text=" ", font=("Courrier", 15), width=10, height=5, bg="SystemButtonFace",
                    command=lambda: self.button_click(8, [b1, b2, b3, b4, b5, b6, b7, b8, b9], pi))
        b9.grid(row=3, column=2)
        root.mainloop()

    def button_click(self, action: int, buttons, pi):
        aa = self.available_actions_ids()
        if action in aa:
            self.act_human_with_action_id(action_id=action, pi=pi)
        self.refresh(buttons)
        if self.is_game_over():
            for i in range(len(buttons)):
                buttons[i].config(state=DISABLED)
            if self.score() == 1.0:
                showinfo("Tic-Tac-Toe", "GagnÃ©")
            elif self.score() == -1.0:
                showinfo("Tic-Tac-Toe", "Perdu")
            else:
                showinfo("Tic-Tac-Toe", "Match null")

    def act_human_with_action_id(self, action_id: int, pi):
        self.current_cell += 3 ** (8 - action_id)
        self.step_count += 1
        if not self.is_game_over():
            s = self.state_id()
            aa = self.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                for a_p in aa:
                    pi[s][a_p] = 1 / len(aa)
            a = np.random.choice(aa, p=[pi[s][a_p] for a_p in aa])
            self.current_cell += 2 * 3 ** (8 - a)
            self.step_count += 1

    def refresh(self, buttons):
        values = self.values_cases()
        for i in range(len(values)):
            if values[i] == 1:
                buttons[i]["text"] = "X"
            elif values[i] == 2:
                buttons[i]["text"] = "O"
        for i in range(3):
            if values[i] == values[i + 3] and values[i + 3] == values[i + (2 * 3)] and (
                    values[i] == 1 or values[i] == 2):
                buttons[i].config(bg="green")
                buttons[i + 3].config(bg="green")
                buttons[i + (2 * 3)].config(bg="green")
            elif values[i * 3] == values[(i * 3) + 1] and values[(i * 3) + 1] == values[(i * 3) + 2] and (
                    values[i * 3] == 1 or values[i * 3] == 2):
                buttons[i * 3].config(bg="green")
                buttons[(i * 3) + 1].config(bg="green")
                buttons[(i * 3) + 2].config(bg="green")
            elif values[i] == values[4] and values[4] == values[8 - i] and (values[i] == 1 or values[i] == 2):
                buttons[i].config(bg="green")
                buttons[4].config(bg="green")
                buttons[8 - i].config(bg="green")


class GridWorld(SingleAgentEnv):
    def __init__(self, cols: int = 5, rows: int = 5):
        self.cols = cols
        self.rows = rows
        self.nb_cells = cols * rows
        self.current_cell = 0
        self.step_count = 0
        self.win_rate = 0.0
        self.game_played = 0

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        if self.step_count > self.nb_cells * 2:
            return True
        return self.current_cell == self.rows - 1 or self.current_cell == self.nb_cells - 1

    def act_with_action_id(self, action_id: int):
        # O: LEFT
        # 1: RIGHT
        # 2: UP
        # 3: DOWN
        self.step_count += 1
        if action_id == 0:
            if self.current_cell % self.rows != 0:
                self.current_cell -= 1
        elif action_id == 1:
            if self.current_cell % self.rows != 4:
                self.current_cell += 1
        elif action_id == 2:
            if self.current_cell - self.rows >= 0:
                self.current_cell -= self.rows
        else:
            if self.current_cell + self.rows <= 24:
                self.current_cell += self.rows

    def score(self) -> float:
        if self.current_cell == self.rows - 1:
            return -3.0
        elif self.current_cell == self.nb_cells - 1:
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.arange(4)

    def reset(self):
        self.current_cell = 0
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for i in range(self.cols):
            for j in range(self.rows):
                if (i * self.rows) + j == self.current_cell:
                    print("X", end='')
                elif i == 0 and j == 0:
                    print("S", end='')
                elif i == 0 and j == (self.rows - 1):
                    print("L", end='')
                elif i == (self.cols - 1) and j == (self.rows - 1):
                    print("W", end='')
                else:
                    print("_", end='')
            print()
        if self.is_game_over():
            self.win_rate += self.score()
            self.game_played += 1
            print(f'win rate : {self.win_rate / self.game_played if self.game_played > 0 else 0}')
            print(f'game played : {self.game_played}')

    def reset_random(self):
        while True:
            self.current_cell = random.randint(0, self.nb_cells - 2)
            if self.current_cell != (self.rows - 1):
                break
        self.step_count = 0


class LineWorld(SingleAgentEnv):
    def __init__(self, nb_cells: int = 5):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)
        self.step_count = 0
        self.win_rate = 0.0
        self.game_played = 0

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        if self.step_count > self.nb_cells * 2:
            return True
        return self.current_cell == 0 or self.current_cell == self.nb_cells - 1

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        if action_id == 0:
            self.current_cell -= 1
        else:
            self.current_cell += 1

    def score(self) -> float:
        if self.current_cell == 0:
            return -1.0
        elif self.current_cell == self.nb_cells - 1:
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.array([0, 1])

    def reset(self):
        self.current_cell = math.floor(self.nb_cells / 2)
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for i in range(self.nb_cells):
            if i == self.current_cell:
                print("X", end='')
            elif i == 0:
                print("L", end='')
            elif i == (self.nb_cells - 1):
                print("W", end='')
            else:
                print("_", end='')
        print()
        if self.is_game_over():
            self.win_rate += self.score()
            self.game_played += 1
            print(f'win rate : {self.win_rate / self.game_played if self.game_played > 0 else 0}')
            print(f'game played : {self.game_played}')

    def reset_random(self):
        self.current_cell = random.randint(1, self.nb_cells - 2)
        self.step_count = 0


class LineWorldMDP(MDPEnv):
    def __init__(self, nb_cells: int = 7):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)

    def states(self) -> np.ndarray:
        return np.arange(self.nb_cells)

    def actions(self) -> np.ndarray:
        return np.array([0, 1])

    def rewards(self) -> np.ndarray:
        return np.array([-1.0, 0.0, 1.0])

    def is_state_terminal(self, s: int) -> bool:
        return s == 0 or s == self.nb_cells - 1

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        if not self.is_state_terminal(s):
            if (s == 1 and a == 0 and s_p == 0 and r == -1.0) or \
                    (s == (self.nb_cells - 2) and a == 1 and s_p == (self.nb_cells - 1) and r == 1.0) or \
                    (a == 0 and s_p == s - 1 and r == 0.0 and s != 1) or \
                    (a == 1 and s_p == s + 1 and r == 0.0 and s != (self.nb_cells - 2)):
                return 1.0
        return 0.0

    def view_state(self, s: int):
        print(f'Game Over: {self.is_state_terminal(s)}')
        if s == 0:
            print("Score : 1.0")
        elif s == (len(self.states()) - 1):
            print("Score : -1.0")
        else:
            print("Score : 0.0")
        for i in range(self.nb_cells):
            if i == s:
                print("X", end='')
            elif i == 0:
                print("L", end='')
            elif i == (len(self.states()) - 1):
                print("W", end='')
            else:
                print("_", end='')
        print()


class GridWorldMDP(MDPEnv):
    def __init__(self, rows: int = 5, cols: int = 5):
        self.nb_cells = rows * cols
        self.rows = rows
        self.cols = cols
        self.current_cell = 0

    def states(self) -> np.ndarray:
        return np.arange(self.nb_cells)

    def actions(self) -> np.ndarray:
        return np.array([0, 1, 2, 3])

    def rewards(self) -> np.ndarray:
        return np.array([-3.0, 0.0, 1.0])

    def is_state_terminal(self, s: int) -> bool:
        return s == (self.rows - 1) or s == (self.nb_cells - 1)

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        if not self.is_state_terminal(s):
            if a == 0:
                if (s % self.rows != 0 and s_p == s - 1 and r == 0.0) or \
                        (s % self.rows == 0 and s_p == s and r == 0.0):
                    return 1.0
            elif a == 1:
                if (s == (self.rows - 2) and s_p == (self.rows - 1) and r == -3.0) or \
                        (s == (self.nb_cells - 2) and s_p == (self.nb_cells - 1) and r == 1.0) or \
                        (s % self.rows == (self.rows - 1) and s_p == s and r == 0.0) or \
                        (s % self.rows != (self.rows - 1) and s_p == s + 1 and r == 0.0 and not self.is_state_terminal(
                            s_p)):
                    return 1.0
            elif a == 2:
                if (s == (self.rows * 2 - 1) and s_p == (self.rows - 1) and r == -3.0) or \
                        (s - self.rows < 0 and s_p == s and r == 0.0) or \
                        (s - self.rows >= 0 and s_p == (s - self.rows) and r == 0.0 and not self.is_state_terminal(
                            s_p)):
                    return 1.0
            else:
                if (s == (self.nb_cells - self.rows - 1) and s_p == (self.nb_cells - 1) and r == 1.0) or \
                        (s + self.rows > (self.nb_cells - 1) and s_p == s and r == 0.0) or \
                        (s + self.rows <= (self.nb_cells - 1) and s_p == (
                                s + self.rows) and r == 0.0 and not self.is_state_terminal(s_p)):
                    return 1.0
        return 0.0

    def view_state(self, s: int):
        print(f'Game Over: {self.is_state_terminal(s)}')
        if s == (self.rows - 1):
            print("Score : 1.0")
        elif s == (len(self.states()) - 1):
            print("Score : -1.0")
        else:
            print("Score : 0.0")
        for i in range(self.cols):
            for j in range(self.rows):
                if (i * self.rows) + j == s:
                    print("X", end='')
                elif i == 0 and j == 0:
                    print("S", end='')
                elif i == 0 and j == (self.rows - 1):
                    print("L", end='')
                elif i == (self.cols - 1) and j == (self.rows - 1):
                    print("W", end='')
                else:
                    print("_", end='')
            print()