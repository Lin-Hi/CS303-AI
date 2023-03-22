import numpy as np
import numba as nb
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(time.time())
TIME_BEFORE_OUT = 0.1


# don't change the class name
# 根据剩余时间选择深度3还是深度5
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        self.start_time = 0.0
        self.not_end_depth_5 = False
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.MAX_DEPTH = 3
        self.weight = np.array([
            [70, -15, 20, 20, 20, 20, -15, 70],
            [-20, -30, 5, 5, 5, 5, -30, -15],
            [20, 5, 1, 1, 1, 1, 5, 20],
            [20, 5, 1, 1, 1, 1, 5, -20],
            [20, 5, 1, 1, 1, 1, 5, -20],
            [20, 5, 1, 1, 1, 1, 5, -20],
            [-20, -30, 5, 5, 5, 5, -30, -15],
            [70, -15, 20, 20, 20, 20, -15, 70]
        ])

    # The input is the current chessboard. Chessboard is a numpy array
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.start_time = time.time()
        self.candidate_list.clear()
        # ==================================================================
        # Find all valid positions here
        # Make sure that the position of your decision on the chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chessboard
        # You need to add all the positions which are valid
        # candidate_list example: [(3,3),(4,4)]
        self.candidate_list = self.get_valid_places(chessboard, self.color)
        # If there is no valid position, you must return an empty list.
        if len(self.candidate_list) == 0:
            return self.candidate_list
        # ==============Find new pos========================================
        self.MAX_DEPTH = 3
        self.not_end_depth_5 = False
        action_dep_3 = self.alpha_beta(chessboard, 0, self.color, float("-inf"), float("inf"))[1]
        self.MAX_DEPTH = 5
        action_dep_5 = self.alpha_beta(chessboard, 0, self.color, float("-inf"), float("inf"))[1]
        if self.not_end_depth_5:
            self.candidate_list.append(action_dep_3)
        else:
            self.candidate_list.append(action_dep_5)
        return self.candidate_list
        # You need append your decision at the end of the candidate_
        # candidate_list example: [(3,3),(4,4),(4,4)]
        # we will pick the last element of the candidate_list as the position you choose.
        # In above example, we will pick (4,4) as your decision.

    def can_go(self, chessboard, color, x, y):
        # (x, y) is COLOR_NONE
        FINDING_OPPO = 0
        FINDING_SELF = 1
        oppo_color = -color
        choice = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in choice:
            tx, ty = x, y
            state = FINDING_OPPO
            while True:
                tx += dx
                ty += dy
                if not (0 <= tx < len(chessboard) and 0 <= ty < len(chessboard[0])):
                    break
                if state == FINDING_OPPO:
                    if chessboard[tx][ty] == oppo_color:
                        state = FINDING_SELF
                    else:
                        break
                else:
                    # state == FINDING_SELF
                    if chessboard[tx][ty] == color:
                        return True
                    if chessboard[tx][ty] == oppo_color:
                        continue
                    if chessboard[tx][ty] == COLOR_NONE:
                        break
        return False

    def calculate(self, chessboard, color):
        count = 0
        oppo_color = -color
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == color:
                    count += self.weight[i][j]
                elif chessboard[i][j] == oppo_color:
                    count -= self.weight[i][j]
        return count

    def get_valid_places(self, chessboard, color):
        idx = np.where(chessboard == COLOR_NONE)
        empty_places = list(zip(idx[0], idx[1]))
        valid_places = []
        for x, y in empty_places:
            if self.can_go(chessboard, color, x, y):
                valid_places.append((x, y))
        return valid_places

    def max_value(self, chessboard, depth, color, a, b):
        if time.time() - self.start_time > self.time_out - 0.2:
            self.not_end_depth_5 = True
            return self.calculate(chessboard, color), None
        if depth > self.MAX_DEPTH:
            depth -= 1
            return self.calculate(chessboard, color), None
        depth += 1
        places = self.get_valid_places(chessboard, color)
        if len(places) == 0:
            return self.calculate(chessboard, color), None
        grades, action = float("-inf"), None
        next_color = -color
        for x, y in places:
            next_chessboard = chessboard.copy()
            next_chessboard[x][y] = next_color
            grades2, _ = self.min_value(next_chessboard, depth, next_color, a, b)
            if grades2 > grades:
                action = x, y
                grades = grades2
            if grades >= b:
                return grades, action
            if grades > a:
                a = grades
        return grades, action

    def min_value(self, chessboard, depth, color, a, b):
        if time.time() - self.start_time > self.time_out - TIME_BEFORE_OUT:
            self.not_end_depth_5 = True
            return self.calculate(chessboard, color), None
        if depth > self.MAX_DEPTH:
            depth -= 1
            return self.calculate(chessboard, color), None
        depth += 1
        places = self.get_valid_places(chessboard, color)
        if len(places) == 0:
            return self.calculate(chessboard, color), None
        grades, action = float("inf"), None
        next_color = -color
        for x, y in places:
            next_chessboard = chessboard.copy()
            next_chessboard[x][y] = next_color
            grades2, _ = self.max_value(next_chessboard, depth, next_color, a, b)
            if grades2 < grades:
                action = x, y
                grades = grades2
            if grades <= a:
                return grades, action
            if grades < b:
                b = grades
        return grades, action

    def alpha_beta(self, chessboard, depth, color, a, b):
        return self.max_value(chessboard, depth, color, a, b)
