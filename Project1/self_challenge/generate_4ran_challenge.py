import numpy as np
import time

from Project1 import randomAI
from Project1 import phase1
from Project1 import phase2
from Project1 import phase3
from Project1 import phase4
from Project1 import phase5

COLOR_BLACK = -1
COLOR_WHITE = 1


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
        # count = 0
        while white_bool and black_bool:
            # t1 = time.time()
            if is_white:
                white_bool = self.run_white()
            else:
                black_bool = self.run_black()
            is_white = not is_white
        #     t2 = time.time()
        #     count += 1
        #     print(str(count) + " " +(str(t2 - t1)))
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
    ran1_win = 0
    ran2_win = 0
    equ = 0
    i = 0
    with open('ran_4_result.txt', 'r') as f:
        arr1 = f.readlines()[-1].split(': ')
        i = int(arr1[0])
        arr2 = arr1[1].split(', ')
        ran1_win = int(arr2[0].split('=')[1])
        ran2_win = int(arr2[1].split('=')[1])
        equ = int(arr2[2].split('=')[1])
        f.close()
    f = open('ran_4_result.txt', 'a')
    try:
        while True:
            ranAI1 = phase4.AI(8, COLOR_WHITE, 5)
            ranAI2 = randomAI.AI(8, COLOR_BLACK, 5)
            auto_machine_1 = Auto_Challenge(AI_white=ranAI1, AI_black=ranAI2)
            result_1 = auto_machine_1.run_game()
            ranAI1 = phase4.AI(8, COLOR_BLACK, 5)
            ranAI2 = randomAI.AI(8, COLOR_WHITE, 5)
            auto_machine_2 = Auto_Challenge(AI_white=ranAI2, AI_black=ranAI1)
            result_2 = -1 * auto_machine_2.run_game()
            i += 1
            if result_1 + result_2 > 0:
                ran1_win += 1
            elif result_1 + result_2 < 0:
                ran2_win += 1
            else:
                equ += 1
            if i % 1 == 0:
                f.write(f"{i}: pha4={ran1_win}, ran={ran2_win}, equ={equ}\n")
            print(f"{i}: pha4={ran1_win}, ran={ran2_win}, equ={equ}")
    except BaseException:
        f.write(f"{i}: ph4={ran1_win}, ran={ran2_win}, equ={equ}\n")
    finally:
        f.flush()
        f.close()
