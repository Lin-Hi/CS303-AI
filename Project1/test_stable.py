import numpy as np
import time
import random

COLOR_NONE = 0


def stable_ans(chessboard, color, stable_list):
    res = [0, 0]
    if chessboard[0][0] != COLOR_NONE:
        color_idx = 0 if chessboard[0][0] == color else 1
        # color_idx == 0: 是我方棋子
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[i][j] == chessboard[0][0]:
                stable_list[i][j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    if chessboard[7][0] != COLOR_NONE:
        color_idx = 0 if chessboard[7][0] == color else 1
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[7 - i][j] == chessboard[7][0]:
                stable_list[7 - i][j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    if chessboard[0][7] != COLOR_NONE:
        color_idx = 0 if chessboard[0][7] == color else 1
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[i][7 - j] == chessboard[0][7]:
                stable_list[i][7 - j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    if chessboard[7][7] != COLOR_NONE:
        color_idx = 0 if chessboard[7][7] == color else 1
        h = 7
        for i in range(8):
            j = 0
            while j <= h and chessboard[7 - i][7 - j] == chessboard[7][7]:
                stable_list[7 - i][7 - j] = True
                res[color_idx] += 1
                j = j + 1
            h = j - 1
            if h == -1:
                break
    return res


def stable_my(chessboard, color, stable_board):
    my_stable = 0
    oppo_stable = 0
    if chessboard[0][0] != COLOR_NONE:
        is_my_chess = chessboard[0][0] == color
        max_width = 7
        line = 0
        while max_width >= 0 and 0 <= line <= 7:
            col = 0
            for col in range(max_width + 1):
                if is_my_chess and chessboard[line][col] == color:
                    stable_board[line][col] = True
                    my_stable += 1
                elif not is_my_chess and chessboard[line][col] == -color:
                    oppo_stable += 1
                    stable_board[line][col] = True
                else:
                    col -= 1
                    break
            max_width = col
            line += 1
    if chessboard[7][0] != COLOR_NONE:
        is_my_chess = chessboard[7][0] == color
        max_width = 7
        line = 7
        while max_width >= 0 and 0 <= line <= 7:
            col = 0
            for col in range(max_width + 1):
                if is_my_chess and chessboard[line][col] == color:
                    my_stable += 1
                    stable_board[line][col] = True
                elif not is_my_chess and chessboard[line][col] == -color:
                    oppo_stable += 1
                    stable_board[line][col] = True
                else:
                    col -= 1
                    break
            max_width = col
            line -= 1
    if chessboard[0][7] != COLOR_NONE:
        is_my_chess = chessboard[0][7] == color
        min_width = 0
        line = 0
        while min_width <= 7 and 0 <= line <= 7:
            col = 7
            for col in range(7, min_width - 1, -1):
                if is_my_chess and chessboard[line][col] == color:
                    my_stable += 1
                    stable_board[line][col] = True
                elif not is_my_chess and chessboard[line][col] == -color:
                    oppo_stable += 1
                    stable_board[line][col] = True
                else:
                    col += 1
                    break
            min_width = col
            line += 1
    if chessboard[7][7] != COLOR_NONE:
        is_my_chess = chessboard[7][7] == color
        min_width = 0
        line = 7
        while min_width <= 7 and 0 <= line <= 7:
            col = 7
            for col in range(7, min_width - 1, -1):
                if is_my_chess and chessboard[line][col] == color:
                    my_stable += 1
                    stable_board[line][col] = True
                elif not is_my_chess and chessboard[line][col] == -color:
                    oppo_stable += 1
                    stable_board[line][col] = True
                else:
                    col += 1
                    break
            min_width = col
            line -= 1
    return (my_stable, oppo_stable)


def random_board():
    random.seed(time.time())
    chessboard = np.zeros((8, 8), dtype='int')
    for i in range(8):
        for j in range(8):
            k = random.randint(1, 10)
            if 1 <= k <= 5:
                chessboard[i][j] = 0
            elif 6 <= k <= 8:
                chessboard[i][j] = 1
            else:
                chessboard[i][j] = -1
    return chessboard


if __name__ == '__main__':
    stable_board_my = [[False for _ in range(8)] for _ in range(8)]
    stable_board_ans = [[False for _ in range(8)] for _ in range(8)]
    # chessboard = np.array([
    #     [0, 0, 0, 0, 0, 0, -1, -1],
    #     [0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0]
    # ])
    for i in range(1000_0000):
        chessboard = random_board()
        ans = stable_ans(chessboard, 1, stable_board_ans)
        my = stable_my(chessboard, 1, stable_board_my)
        if not (ans == list(my) and stable_board_ans == stable_board_my):
            print(chessboard)
            print(ans)
            print(my)
            break
        if (i + 1) % 1_0000 == 0:
            print(i + 1)
    print('done')
