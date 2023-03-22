import numpy as np
import numba as nb
import random
import time

INF = 9999999999
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(time.time())
TIME_BEFORE_OUT = 0.1
Dir = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class TestLocation:
    def __init__(self, color):
        self.color = color
        self.chessboard_size = 8
        self.weight_1_2 = np.array([
            [39, -1, 12, 38, 38, 12, -1, 39],
            [-1, -47, 26, 21, 21, 26, -47, -1],
            [12, 26, 44, -59, -59, 44, 26, 12],
            [38, 21, -59, -44, -44, -59, 21, 38],
            [38, 21, -59, -44, -44, -59, 21, 38],
            [12, 26, 44, -59, -59, 44, 26, 12],
            [-1, -47, 26, 21, 21, 26, -47, -1],
            [39, -1, 12, 38, 38, 12, -1, 39]
        ])
        self.weight_3 = np.array([
            [45, 25, 56, 33, 33, 56, 25, 45],
            [25, 42, 61, 18, 18, 61, 42, 25],
            [56, 61, -11, 45, 45, -11, 61, 56],
            [33, 18, 45, -48, -48, 45, 18, 33],
            [33, 18, 45, -48, -48, 45, 18, 33],
            [56, 61, -11, 45, 45, -11, 61, 56],
            [25, 42, 61, 18, 18, 61, 42, 25],
            [45, 25, 56, 33, 33, 56, 25, 45]
        ])
        self.mobility_1_2 = np.array([
            [42, 12, 39, 10, 10, 39, 12, 42],
            [12, -10, -50, 28, 28, -50, -10, 12],
            [39, -50, 3, 44, 44, 3, -50, 39],
            [10, 28, 44, -53, -53, 44, 28, 10],
            [10, 28, 44, -53, -53, 44, 28, 10],
            [39, -50, 3, 44, 44, 3, -50, 39],
            [12, -10, -50, 28, 28, -50, -10, 12],
            [42, 12, 39, 10, 10, 39, 12, 42]
        ])
        self.mobility_3 = np.array([
            [22, 57, -55, 49, 49, -55, 57, 22],
            [57, -32, 1, -1, -1, 1, -32, 57],
            [-55, 1, -57, -36, -36, -57, 1, -55],
            [49, -1, -36, -64, -64, -36, -1, 49],
            [49, -1, -36, -64, -64, -36, -1, 49],
            [-55, 1, -57, -36, -36, -57, 1, -55],
            [57, -32, 1, -1, -1, 1, -32, 57],
            [22, 57, -55, 49, 49, -55, 57, 22]
        ])

    def stable(self, chessboard: list, stable_list: list):
        res = [0, 0]
        if chessboard[0][0] != COLOR_NONE:
            color_idx = 0 if chessboard[0][0] == self.color else 1
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
            color_idx = 0 if chessboard[7][0] == self.color else 1
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
            color_idx = 0 if chessboard[0][7] == self.color else 1
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
            color_idx = 0 if chessboard[7][7] == self.color else 1
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

    def evaluation(self, chessboard):
        """
        Evaluate a static situation.

        :param chessboard: the current chessboard
        :return: the evaluation value
        """
        # res0 = self.match.get(board_hash(chessboard, current_pos))
        # if res0 is not None:
        #     return res0
        initial, stable, arb, front = 0, 0, 0, 0
        # 权值和   稳定子  行动力 边缘子
        is_stable = [[False for _ in range(8)] for _ in range(8)]
        sc, my = 0, 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] != COLOR_NONE:
                    sc += 1
                if chessboard[i][j] == self.color:
                    my += 1
        # sc: total chess number, my: this color number
        arb_my = self.get_valid_places(chessboard, self.color)
        # 我可以下子的地方
        arb_opp = self.get_valid_places(chessboard, -self.color)
        # 对方可以下子的地方

        if len(arb_my) == 0 and len(arb_opp) == 0:
            # 判断终局
            return INF if my < sc // 2 else (0 if my == sc // 2 else -INF)

        stable_my, stable_opp = self.stable(chessboard, is_stable)
        # 双方的稳定子
        stable = (stable_opp - stable_my)
        diff = my + my - sc
        # 我的棋子数 - 他的棋子数
        k = (sc - 5) // 20
        # 5~24： 开局
        # 25~44： 中局
        # 45~64： 残局
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                # 算 8x8 位置权重
                if chessboard[i][j] == COLOR_NONE:
                    continue
                if sc <= 44:
                    initial += self.weight_1_2[i][j] if chessboard[i][j] != self.color else -self.weight_1_2[i][
                        j]
                else:
                    initial += self.weight_3[i][j] if chessboard[i][j] != self.color else - \
                        self.weight_3[i][j]

        # 算行动力，也用 8x8 数组
        for action in arb_my:
            arb += self.mobility_1_2[action[0]][action[1]] if sc <= 44 else self.mobility_3[action[0]][action[1]]
        for action in arb_opp:
            arb -= self.mobility_1_2[action[0]][action[1]] if sc <= 44 else self.mobility_3[action[0]][action[1]]

        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                # 暴力找边缘子
                if chessboard[i][j] != COLOR_NONE and not is_stable[i][j]:
                    # Dir: 八个方向的 list
                    for kk in Dir:
                        a = i + kk[0]
                        b = j + kk[1]
                        if 0 <= a <= 7 and 0 <= b <= 7 and chessboard[a][b] == COLOR_NONE:
                            front += 1 if chessboard[i][j] == self.color else -1

        c1 = 1
        c2 = [48, 107, 97]
        c3 = [-33, -60, -7]
        c4 = [-18, -41, -112]
        result = c1 * initial + c2[k] * stable + c3[k] * diff + c4[k] * front + arb
        return result

    def get_stable_disc(self, chessboard, stable_board):
        my_stable = 0
        oppo_stable = 0
        if chessboard[0][0] != COLOR_NONE:
            is_my_chess = chessboard[0][0] == self.color
            max_width = 7
            line = 0
            while max_width >= 0 and 0 <= line <= 7:
                col = 0
                for col in range(max_width + 1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        stable_board[line][col] = True
                        my_stable += 1
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col -= 1
                        break
                max_width = col
                line += 1
        if chessboard[7][0] != COLOR_NONE:
            is_my_chess = chessboard[7][0] == self.color
            max_width = 7
            line = 7
            while max_width >= 0 and 0 <= line <= 7:
                col = 0
                for col in range(max_width + 1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        my_stable += 1
                        stable_board[line][col] = True
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col -= 1
                        break
                max_width = col
                line -= 1
        if chessboard[0][7] != COLOR_NONE:
            is_my_chess = chessboard[0][7] == self.color
            min_width = 0
            line = 0
            while min_width <= 7 and 0 <= line <= 7:
                col = 7
                for col in range(7, min_width - 1, -1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        my_stable += 1
                        stable_board[line][col] = True
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col += 1
                        break
                min_width = col
                line += 1
        if chessboard[7][7] != COLOR_NONE:
            is_my_chess = chessboard[7][7] == self.color
            min_width = 0
            line = 7
            while min_width <= 7 and 0 <= line <= 7:
                col = 7
                for col in range(7, min_width - 1, -1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        my_stable += 1
                        stable_board[line][col] = True
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col += 1
                        break
                min_width = col
                line -= 1
        return (my_stable, oppo_stable)

    def calculate(self, chessboard):

        # 判断前中后状态
        my_num = 0
        oppo_num = 0
        for line in chessboard:
            for chess in line:
                if chess == self.color:
                    my_num += 1
                elif chess == -self.color:
                    oppo_num += 1
        total_num = my_num + oppo_num
        diff_num = my_num - oppo_num
        state = 0
        # 0：未赋值     1： 开局     2：中局     3：残局
        if total_num < 25:
            state = 1
        elif total_num < 45:
            state = 2
        else:
            state = 3

        # 判断终局
        valid_my = self.get_valid_places(chessboard, self.color)
        valid_oppo = self.get_valid_places(chessboard, -self.color)
        if len(valid_my) == 0 and len(valid_oppo) == 0:
            if my_num < oppo_num:
                return 9999999999
            elif my_num > oppo_num:
                return -9999999999
            else:
                return 0

        # 求权值和
        location_weight = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if state != 3:
                    if chessboard[i][j] == self.color:
                        location_weight -= self.weight_1_2[i][j]
                    elif chessboard[i][j] == -self.color:
                        location_weight += self.weight_1_2[i][j]
                else:
                    if chessboard[i][j] == self.color:
                        location_weight -= self.weight_3[i][j]
                    elif chessboard[i][j] == -self.color:
                        location_weight += self.weight_3[i][j]
        # location_weight 计算完毕

        # 算行动力，用 8x8 数组表示行动力
        mobility = 0
        valid_my = self.get_valid_places(chessboard, self.color)
        valid_oppo = self.get_valid_places(chessboard, -self.color)
        for x, y in valid_my:
            if state != 3:
                mobility += self.mobility_1_2[x][y]
            else:
                mobility += self.mobility_3[x][y]
        for x, y in valid_oppo:
            if state != 3:
                mobility -= self.mobility_1_2[x][y]
            else:
                mobility -= self.mobility_3[x][y]
        # mobility 计算完毕

        # 计算双方稳定子和差值
        stable_board = [
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]
        ]
        stable_my, stable_oppo = self.get_stable_disc(chessboard, stable_board)
        stable = stable_oppo - stable_my
        # stable 计算完毕

        # 计算边缘子
        frontier_my = 0
        frontier_oppo = 0
        direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] == COLOR_NONE or stable_board[i][j]:
                    continue
                for dx, dy in direction:
                    x = i + dx
                    y = j + dy
                    if 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == COLOR_NONE:
                        if chessboard[i][j] == self.color:
                            frontier_my += 1
                        else:
                            frontier_oppo += 1
        front = frontier_my - frontier_oppo
        # front 计算完毕

        c1 = 1
        c2 = [None, 48, 107, 97]
        c3 = [None, -33, -60, -7]
        c4 = [None, -18, -41, -112]
        result = c1 * location_weight + \
                 c2[state] * stable + \
                 c3[state] * diff_num + \
                 c4[state] * front + \
                 mobility
        return result


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
    count = 0
    chessboard = np.zeros((8, 8), dtype='int')
    f = open('chess_log.txt', 'r')
    for s in f.readlines():
        s = s.replace('[','').replace(']','')
        arr = s.split(',')
        for i in arr:
            chessboard[count // 8][count % 8] = int(i)
            count += 1
            if count == 64:
                test_location = TestLocation(COLOR_WHITE)
                ans = test_location.evaluation(chessboard)
                my = test_location.calculate(chessboard)
                count = 0
                chessboard = np.zeros((8, 8), dtype='int')
                if ans != my:
                    print('error')
                    exit(1)
    print(1)
