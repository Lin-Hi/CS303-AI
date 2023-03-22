import random

import numpy as np
import time
import sys

from Project1 import randomAI
from Project1 import phase1
from Project1 import phase2
from Project1 import phase3
from Project1 import phase4
from Project1 import phase5
from Project1 import phase6
from Project1 import phase7

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


class randomAI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []

    def go(self, chessboard):
        self.candidate_list.clear()
        self.candidate_list = self.get_valid_places(chessboard, self.color)
        if len(self.candidate_list) == 0:
            return self.candidate_list
        i = random.randint(0, len(self.candidate_list) - 1)
        self.candidate_list.append(self.candidate_list[i])
        return self.candidate_list

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

    def get_valid_places(self, chessboard, color):
        idx = np.where(chessboard == COLOR_NONE)
        empty_places = list(zip(idx[0], idx[1]))
        valid_places = []
        for x, y in empty_places:
            if self.can_go(chessboard, color, x, y):
                valid_places.append((x, y))
        return valid_places


class Auto_Challenge(object):
    def __init__(self, AI_black, AI_white):
        self.AI_black = AI_black
        self.AI_white = AI_white
        self.candidate_list_1 = [(-1, -1)]
        self.candidate_list_2 = [(-1, -1)]
        self.start_time = 0
        self.end_time = 0
        self.chessboard = np.zeros((8, 8), dtype='int64')
        self.chessboard[3][3] = COLOR_WHITE
        self.chessboard[4][4] = COLOR_WHITE
        self.chessboard[4][3] = COLOR_BLACK
        self.chessboard[3][4] = COLOR_BLACK

    def initialize_chessboard(self):
        self.chessboard = np.zeros((8, 8), dtype='int64')
        self.chessboard[3][3] = COLOR_WHITE
        self.chessboard[4][4] = COLOR_WHITE
        self.chessboard[4][3] = COLOR_BLACK
        self.chessboard[3][4] = COLOR_BLACK

    def run_black(self):
        self.candidate_list_1 = self.AI_black.go(self.chessboard)
        if len(self.candidate_list_1) != 0 and len(self.candidate_list_2) != 0:
            self.update_chessboard(self.candidate_list_1[-1], COLOR_BLACK)
            return True
        return False

    def run_white(self):
        self.candidate_list_2 = self.AI_white.go(self.chessboard)
        if len(self.candidate_list_1) != 0 and len(self.candidate_list_2) != 0:
            self.update_chessboard(self.candidate_list_2[-1], COLOR_WHITE)
            return True
        return False

    def start_game(self):
        self.start_time = time.time()

    def end_game(self):
        self.end_time = time.time()
        run_time = self.end_time - self.start_time
        white_num = 0
        black_num = 0
        for line in self.chessboard:
            for chess in line:
                if chess == COLOR_WHITE:
                    white_num += 1
                else:
                    black_num += 1
        self.initialize_chessboard()
        if white_num < black_num:
            return COLOR_WHITE
        elif white_num > black_num:
            return COLOR_BLACK
        else:
            return 0
        # print(run_time)
        # print(self.chessboard)

    def run_game(self):
        self.start_game()
        white_bool = True
        black_bool = True
        is_white = True
        count = 0
        while white_bool and black_bool:
            t1 = time.time()
            if is_white:
                white_bool = self.run_white()
            else:
                black_bool = self.run_black()
            t2 = time.time()
            if is_white:
                print(str(count) + " " + (str(t2 - t1)))
            is_white = not is_white
            count += 1
        return self.end_game()

    def update_chessboard(self, pos, color):
        steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        x, y = pos
        self.chessboard[x][y] = color
        for dx, dy in steps:
            self.update_chessboard_line(x, y, dx, dy, color)

    def update_chessboard_line(self, x, y, dx, dy, color):
        x_ = x + dx
        y_ = y + dy
        passed = []
        while 0 <= x_ < 8 and 0 <= y_ < 8:
            if self.chessboard[x_][y_] == -color:
                passed.append((x_, y_))
                x_ += dx
                y_ += dy
                continue
            if self.chessboard[x_][y_] == 0:
                break
            if self.chessboard[x_][y_] == color:
                for i, j in passed:
                    self.chessboard[i][j] = color
                break


if __name__ == '__main__':
    log_print = open('5_7_result.txt', 'a')
    sys.stdout = log_print
    sys.stderr = log_print
    i = 0
    while True:
        random.seed(i)
        print('phase5: (3 -> 5)')
        ranAI1 = phase5.AI(8, COLOR_WHITE, 5)
        ranAI2 = randomAI(8, COLOR_BLACK, 5)
        auto_machine_1 = Auto_Challenge(AI_white=ranAI1, AI_black=ranAI2)
        result_1 = auto_machine_1.run_game()
        random.seed(i)
        print('\nphase7: (1 -> 3 -> 5)')
        ranAI2 = phase7.AI(8, COLOR_WHITE, 5)
        ranAI1 = randomAI(8, COLOR_BLACK, 5)
        auto_machine_2 = Auto_Challenge(AI_white=ranAI2, AI_black=ranAI1)
        result_2 = -1 * auto_machine_2.run_game()
        i += 1
