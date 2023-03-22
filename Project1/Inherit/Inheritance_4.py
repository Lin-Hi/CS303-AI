import multiprocessing
from multiprocessing import Pool
import re
import numpy.random
from queue import Queue
from numba import njit
times = 200
import math
import numpy as np
import random
import time

bigInf = 10000000
smallInf = 10000000
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed()
playNum = 55
waiter_num = 56
INF = 999999

order = (
(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),
(7, 7),
(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
(2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
(2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),
(3, 2), (4, 2), (3, 5), (4, 5),
(3, 3), (3, 4), (4, 3), (4, 4),
)

order2 = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
    (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
    (2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),
    (3, 2), (4, 2), (3, 5), (4, 5),
    (3, 3), (3, 4), (4, 3), (4, 4),
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
    (7, 6), (7, 7),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
)

order3 = (
    (3, 3), (3, 4), (4, 3), (4, 4),

    (3, 2), (4, 2), (3, 5), (4, 5),
    (2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),

    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
    (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),

    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
    (7, 6), (7, 7),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
)

order4 = (
    (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)
)

order5 = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)
)

dir = [[1, 0],
       [-1, 0],
       [0, 1],
       [0, -1],
       [1, 1],
       [-1, - 1],
       [1, -1],
       [-1, 1]]
corner=((0,0),(0,7),(7,0),(7,7))
corner_neighbor=((0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6))

class Node :
    def __init__(self) :
        self.arrivable = []
        self.n = 0
        self.v = 0
        self.sub_nodes = []

time_shoot = 1234567

order0 = [(6, 5), (4, 6), (2, 6), (4, 1), (1, 1), (3, 6), (6, 6), (1, 3), (1, 4), (6, 4), (1, 6), (6, 2),
          (1, 5), (5, 6),
          (6, 1), (5, 1), (6, 3), (1, 2), (3, 1), (2, 1),
          (5, 4), (3, 2), (4, 2), (2, 5), (2, 4), (2, 2), (4, 5), (5, 5), (5, 3), (3, 5), (2, 3), (5, 2),
          (3, 3), (4, 3), (3, 4), (4, 4),
          (1, 0), (1, 7), (0, 5), (4, 0), (7, 0), (0, 4), (0, 6), (2, 7), (3, 0), (7, 6), (7, 1), (0, 3),
          (6, 7), (7, 5),
          (7, 3), (5, 7), (0, 7), (7, 2), (7, 7), (3, 7), (5, 0), (6, 0), (7, 4), (0, 2), (0, 0), (2, 0),
          (4, 7), (0, 1)]

winner_order = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5),
    (7, 6), (7, 7),
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
    (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
    (2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),
    (3, 2), (4, 2), (3, 5), (4, 5),
    (3, 3), (3, 4), (4, 3), (4, 4)]

Dir = [(-1, -1), (1, -1), (-1, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (0, 1)]

Dir2 = [(1, 0), (0, -1), (-1, 0), (0, 1)]

def calculate(stat: list, dp_array):
    stat_hash: int = 0
    pow3: int = 1
    for i in range(8):
        stat_hash += (stat[i] + 1) * pow3
        pow3 *= 3
    if dp_array[stat_hash] > -9999:
        return dp_array[stat_hash]
    zero = 0
    for _ in range(8):
        zero += 1 if stat[_] == COLOR_NONE else 0
    res = 0.0
    stat_0 = [0 for _ in range(8)]
    for j in range(8):
        stat_0[j] = stat[j]
    if zero == 0:
        res += (stat[0] + stat[7]) * 2
        for i in range(1, 7):
            res += stat[i]
    else:
        for i in range(8):
            if stat[i] == COLOR_NONE:
                flag = 0
                l_, r_ = i - 1, i + 1
                l_list, r_list = [], []
                if l_ >= 0 and stat[l_] != COLOR_NONE:
                    color_l = stat[l_]
                    while l_ >= 0 and stat[l_] == color_l:
                        l_ -= 1
                    if l_ >= 0 and l_ != i - 1 and stat[l_] == -color_l:
                        flag += color_l
                        for j in range(l_ + 1, i):
                            l_list.append(j)
                if r_ < 8 and stat[r_] != COLOR_NONE:
                    color_r = stat[r_]
                    while r_ < 8 and stat[r_] == color_r:
                        r_ += 1
                    if r_ < 8 and r_ != i + 1 and stat[r_] == -color_r:
                        flag += color_r
                        for j in range(i + 1, r_):
                            r_list.append(j)

                stat[i] = -1
                flag_l, flag_r = False, False
                if len(l_list) > 0 and stat[l_list[0]] == 1:
                    flag_l = True
                    for j in l_list:
                        stat[j] = -1
                if len(r_list) > 0 and stat[r_list[0]] == 1:
                    flag_r = True
                    for j in r_list:
                        stat[j] = -1
                res += calculate(stat, dp_array) * 1 / zero * (1 / 2 - flag * 1 / 6)
                if flag_l:
                    for j in l_list:
                        stat[j] = 1
                if flag_r:
                    for j in r_list:
                        stat[j] = 1

                flag_l, flag_r = False, False
                stat[i] = 1
                if len(l_list) > 0 and stat[l_list[0]] == -1:
                    flag_l = True
                    for j in l_list:
                        stat[j] = 1
                if len(r_list) > 0 and stat[r_list[0]] == -1:
                    flag_r = True
                    for j in r_list:
                        stat[j] = 1
                res += calculate(stat, dp_array) * 1 / zero * (1 / 2 + flag * 1 / 6)
                if flag_l:
                    for j in l_list:
                        stat[j] = -1
                if flag_r:
                    for j in r_list:
                        stat[j] = -1
                stat[i] = 0
    dp_array[stat_hash] = res
    return res


def bit_board(chessboard) -> tuple:
    a, b = 0, 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == COLOR_BLACK:
                a |= (1 << (i * 8 + j))
            elif chessboard[i][j] == COLOR_WHITE:
                b |= (1 << (i * 8 + j))
    return a, b


def legal_(p):
    return (0 <= p[0] < 8) and (0 <= p[1] < 8)


def get_accessible_(chessboard: np.array, turn, pos) -> bool:
    if chessboard[pos[0]][pos[1]] != COLOR_NONE:
        return False
    for k in range(8):
        p = (pos[0] + Dir[k][0], pos[1] + Dir[k][1])
        while legal_(p) and chessboard[p[0]][p[1]] == -turn:
            p = (p[0] + Dir[k][0], p[1] + Dir[k][1])
        if legal_(p) and p != (pos[0] + Dir[k][0], pos[1] + Dir[k][1]) and chessboard[p[0]][
            p[1]] == turn:
            return True
    return False


def get_next_(chessboard: np.array, turn, pos):
    res, res_tmp = [], []
    for k in range(8):
        res_tmp.clear()
        p = (pos[0] + Dir[k][0], pos[1] + Dir[k][1])
        while legal_(p) and chessboard[p[0]][p[1]] == -turn:
            res_tmp.append(p)
            p = (p[0] + Dir[k][0], p[1] + Dir[k][1])
        if legal_(p) and p != (pos[0] + Dir[k][0], pos[1] + Dir[k][1]) and chessboard[p[0]][p[1]] == turn:
            res += res_tmp
    for pos0 in res:
        chessboard[pos0[0]][pos0[1]] = turn
    chessboard[pos[0]][pos[1]] = turn


def get_accessible_list_(chessboard: np.array, turn) -> list:
    """
    Get a list that contains all valid position.

    :param chessboard: the current chessboard
    :param turn: the color of the side to be dropped
    :return: the list
    """
    lis = []
    for i in range(8):
        for j in range(8):
            if get_accessible_(chessboard, turn, (i, j)):
                lis.append((i, j))
    return lis


@njit()
def calculate0(s0, s1, s2, s3, s4, s5, s6, s7, dp_array):
    x0: int = (s0 + 1) + (s1 + 1) * 3 + (s2 + 1) * 9 + (s3 + 1) * 27 + (s4 + 1) * 81 + (s5 + 1) * 243 + (
            s6 + 1) * 729 + (s7 + 1) * 2187
    return dp_array[x0]


@njit()
def num_chess(chessboard: np.array, turn) -> tuple:
    """
    number of chess in a chessboard.

    :param turn: who is playing
    :param chessboard: the current chessboard
    :return: an integer
    """
    sc = 0
    my = 0
    for i in range(8):
        for j in range(8):
            sc += (1 if chessboard[i][j] != COLOR_NONE else 0)
            my += (1 if chessboard[i][j] == turn else 0)
    return sc, my


@njit()
def legal(p):
    """
    To check whether position p is out of bounds.

    :param p: position
    :return: whether p is out-of-bound
    """
    return (0 <= p[0] < 8) and (0 <= p[1] < 8)


@njit()
def get_accessible(chessboard: np.array, now_turn, pos, Dir0: np.array) -> bool:
    """
    Check if the current move at pos is legal.

    :param Dir0:
    :param chessboard: the current chessboard
    :param now_turn: the color of the side that will play
    :param pos: the position to be dropped
    :return: whether pos is legal
    """
    if chessboard[pos[0]][pos[1]] != COLOR_NONE:
        return False
    for kk in Dir0:
        p = (pos[0] + kk[0], pos[1] + kk[1])
        while legal(p) and chessboard[p[0]][p[1]] == -now_turn:
            p = (p[0] + kk[0], p[1] + kk[1])
        if legal(p) and p != (pos[0] + kk[0], pos[1] + kk[1]) and chessboard[p[0]][p[1]] == now_turn:
            return True
    return False


@njit()
def next_board(chessboard: np.array, pos, now_turn, Dir0: np.array) -> list:
    """
    Return the next chessboard when a move is made
    We default that pos is legal. Otherwise, the function will have unpredictable behavior.

    :param Dir0:
    :param chessboard: the current chessboard(will be changed to the next chessboard)
    :param now_turn: the color of the side that will play
    :param pos: the position to be dropped
    :return:
    """
    res, res_tmp = [(-1, -1)], [(-1, -1)]
    res.clear()
    for kk in Dir0:
        res_tmp.clear()
        p = (pos[0] + kk[0], pos[1] + kk[1])
        while legal(p) and chessboard[p[0]][p[1]] == -now_turn:
            res_tmp.append(p)
            p = (p[0] + kk[0], p[1] + kk[1])
        if legal(p) and p != (pos[0] + kk[0], pos[1] + kk[1]) and chessboard[p[0]][p[1]] == now_turn:
            res += res_tmp
    for pos0 in res:
        chessboard[pos0[0]][pos0[1]] = now_turn
    chessboard[pos[0]][pos[1]] = now_turn
    return res


@njit()
def get_accessible_list(chessboard: np.array, now_turn, Dir0: np.array) -> list:
    """
    Get a list that contains all valid position.

    :param Dir0:
    :param chessboard: the current chessboard
    :param now_turn: the color of the side to be dropped
    :return: the list
    """
    lis = [(-1, -1)]
    lis.clear()
    for i in range(8):
        for j in range(8):
            if get_accessible(chessboard, now_turn, (i, j), Dir0):
                lis.append((i, j))
    return lis


@njit()
def stable_calc(chessboard: np.array, stable_list: np.array, self_color):
    res = [0, 0]
    if chessboard[0][0] != COLOR_NONE:
        color_idx = 0 if chessboard[0][0] == self_color else 1
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
        color_idx = 0 if chessboard[7][0] == self_color else 1
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
        color_idx = 0 if chessboard[0][7] == self_color else 1
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
        color_idx = 0 if chessboard[7][7] == self_color else 1
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


@njit()
def initial_calc(chessboard: np.array, score_matrix: np.array, score_matrix1: np.array, self_color, numC):
    initial = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == COLOR_NONE:
                continue
            if numC <= 44:
                initial += score_matrix[i][j] if chessboard[i][j] != self_color else -score_matrix[i][j]
            else:
                initial += score_matrix1[i][j] if chessboard[i][j] != self_color else -score_matrix1[i][j]
    return initial


@njit()
def front_calc(chessboard: np.array, is_stable: np.array, self_color, Dir0: np.array):
    front = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] != COLOR_NONE and not is_stable[i][j]:
                for kk in Dir0:
                    a = i + kk[0]
                    b = j + kk[1]
                    if legal((a, b)) and chessboard[a][b] == COLOR_NONE:
                        front += 1 if chessboard[i][j] == self_color else -1
    return front


@njit()
def arb_calc(chessboard: np.array, self_color, arb_matrix: np.array, arb_matrix1: np.array, sc, dir0: np.array):
    arb = 0
    arb_my = get_accessible_list(chessboard, self_color, dir0)
    arb_opp = get_accessible_list(chessboard, -self_color, dir0)
    for action in arb_my:
        arb += arb_matrix[action[0]][action[1]] if sc <= 44 else arb_matrix1[action[0]][action[1]]
    for action in arb_opp:
        arb -= arb_matrix[action[0]][action[1]] if sc <= 44 else arb_matrix1[action[0]][action[1]]
    return arb


@njit()
def final_search(Order: np.array, dir0: np.array, chessboard: np.array, turn, deep: int, last_chance) -> tuple:
    if deep == 0:
        return time_shoot, -1, -1
    result = -INF if (deep & 1) == 0 else INF
    any_pos_flag = False
    final_pos_x, final_pos_y = -1, -1
    for ite in Order:
        if get_accessible(chessboard, turn, ite, dir0):
            any_pos_flag = True
            # movement
            changed = next_board(chessboard, ite, turn, dir0)
            nxt = final_search(Order, dir0, chessboard, -turn, deep - 1, True)
            chessboard[ite[0]][ite[1]] = COLOR_NONE
            for _ in range(len(changed)):
                xx, yy = changed[_][0], changed[_][1]
                chessboard[xx][yy] = -turn
            if nxt[0] > 1234560:
                if (deep & 1) == 0:
                    return result, -1, -1
                else:
                    return -INF, -1, -1
            if (deep & 1) == 0:
                if result < nxt[0]:
                    result = nxt[0]
                    final_pos_x, final_pos_y = ite
                    if result == INF:
                        if deep == 100:
                            break
                        return result, -1, -1
            else:
                if result > nxt[0]:
                    result = nxt[0]
                if result == -INF:
                    return result, -1, -1
    # At the first layer, returns the position
    if deep == int(100):
        return result, final_pos_x, final_pos_y
    # There is no location to go down (at this time to ensure that it will not appear in the first layer)
    elif not any_pos_flag:
        sc, my = num_chess(chessboard, turn if (deep & 1) == 0 else -turn)
        if sc == 64 or not last_chance:
            return INF if my < sc // 2 else (0 if my == sc // 2 else -INF), -1, -1
        else:
            return final_search(Order, dir0, chessboard, -turn, deep - 1, False)
    return result, final_pos_x, final_pos_y


class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size

        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list.
        # The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.maxDepth = 1
        self.single_time_out = 0
        self.final_time_out = 0
        self.is_first_round = 2
        self.final_search_begin = 52
        self.history_black_win = 0
        self.history_white_win = 0
        self.live_time = 0
        self.win_num = 0
        self.white_win = 0
        self.black_win = 0
        self.tie_num = 0

        self.score_matrix = np.array(
            [[-33, -31, 57, 62, 62, 57, -31, -33], [-31, -16, 62, 2, 2, 62, -16, -31],
             [57, 62, -42, -55, -55, -42, 62, 57],
             [62, 2, -55, -53, -53, -55, 2, 62], [62, 2, -55, -53, -53, -55, 2, 62],
             [57, 62, -42, -55, -55, -42, 62, 57],
             [-31, -16, 62, 2, 2, 62, -16, -31], [-33, -31, 57, 62, 62, 57, -31, -33]])
        self.score_matrix1 = np.array(
            [[-52, -64, -1, -42, -42, -1, -64, -52], [-64, 56, 27, 34, 34, 27, 56, -64],
             [-1, 27, 12, 30, 30, 12, 27, -1],
             [-42, 34, 30, -1, -1, 30, 34, -42], [-42, 34, 30, -1, -1, 30, 34, -42], [-1, 27, 12, 30, 30, 12, 27, -1],
             [-64, 56, 27, 34, 34, 27, 56, -64], [-52, -64, -1, -42, -42, -1, -64, -52]])
        self.arb_matrix = np.array(
            [[63, 3, 27, -18, -18, 27, 3, 63], [3, 23, -48, -64, -64, -48, 23, 3], [27, -48, 49, 5, 5, 49, -48, 27],
             [-18, -64, 5, 11, 11, 5, -64, -18], [-18, -64, 5, 11, 11, 5, -64, -18], [27, -48, 49, 5, 5, 49, -48, 27],
             [3, 23, -48, -64, -64, -48, 23, 3], [63, 3, 27, -18, -18, 27, 3, 63]])
        self.arb_matrix1 = np.array(
            [[21, 50, 34, 14, 14, 34, 50, 21], [50, -17, 52, 50, 50, 52, -17, 50], [34, 52, 24, 22, 22, 24, 52, 34],
             [14, 50, 22, -31, -31, 22, 50, 14], [14, 50, 22, -31, -31, 22, 50, 14], [34, 52, 24, 22, 22, 24, 52, 34],
             [50, -17, 52, 50, 50, 52, -17, 50], [21, 50, 34, 14, 14, 34, 50, 21]])
        self.c1, self.c2, self.c3, self.c4 = 1, [23, 184, 233], [-39, -54, -40], [-27, -53, -49]

        self.order_x = [(6, 5), (4, 6), (2, 6), (4, 1), (1, 1), (3, 6), (6, 6), (1, 3), (1, 4), (6, 4), (1, 6), (6, 2),
                        (1, 5), (5, 6),
                        (6, 1), (5, 1), (6, 3), (1, 2), (3, 1), (2, 1),
                        (5, 4), (3, 2), (4, 2), (2, 5), (2, 4), (2, 2), (4, 5), (5, 5), (5, 3), (3, 5), (2, 3), (5, 2),
                        (3, 3), (4, 3), (3, 4), (4, 4),
                        (1, 0), (1, 7), (0, 5), (4, 0), (7, 0), (0, 4), (0, 6), (2, 7), (3, 0), (7, 6), (7, 1), (0, 3),
                        (6, 7), (7, 5),
                        (7, 3), (5, 7), (0, 7), (7, 2), (7, 7), (3, 7), (5, 0), (6, 0), (7, 4), (0, 2), (0, 0), (2, 0),
                        (4, 7), (0, 1)]
        self.dir = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (0, 1)])
        self.dp_array = np.array([-9999.999 for _ in range(6561)])  # 3^8
        self.winner_order = np.array([
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),
            (7, 5),
            (7, 6), (7, 7),
            (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
            (2, 1), (3, 1), (4, 1), (5, 1), (2, 6), (3, 6), (4, 6), (5, 6),
            (2, 2), (2, 3), (2, 4), (2, 5), (5, 2), (5, 3), (5, 4), (5, 5),
            (3, 2), (4, 2), (3, 5), (4, 5),
            (3, 3), (3, 4), (4, 3), (4, 4)])
        self.kjk = {(34628173824, 68853694464): (2, 3), (34762915840, 68719476736): (4, 2),
                    (34829500416, 68719476736): (2, 4), (240786604032, 134217728): (5, 3),
                    (17695533694976, 134217728): (3, 5)}
        self.current_num = 0

    def evaluation(self, chessboard: np.array):
        """
        Evaluate a static situation.

        :param chessboard: the current chessboard
        :return: the evaluation value
        """
        # res0 = self.match.get(board_hash(chessboard, current_pos))
        # if res0 is not None:
        #     return res0
        is_stable = np.zeros((8, 8), dtype=bool)
        sc, my = num_chess(chessboard, self.color)
        stable_my, stable_opp = stable_calc(chessboard, is_stable, self.color)
        stable = (stable_opp - stable_my) * (1 + 0.2 * max(stable_opp, stable_my))
        diff = my + my - sc
        k = (sc - 5) // 20
        initial = initial_calc(chessboard, self.score_matrix, self.score_matrix1, self.color, sc)
        arb = arb_calc(chessboard, self.color, self.arb_matrix, self.arb_matrix1, sc, self.dir)
        front = front_calc(chessboard, is_stable, self.color, self.dir)
        result = self.c1 * initial + self.c2[k] * stable + self.c3[k] * diff + self.c4[k] * front + arb
        return result

    def ab(self, chessboard: np.array, turn, deep, alpha, beta, last_chance, timeout) -> tuple:
        # if self.current_num + self.maxDepth - deep >= 56 and turn == self.color:
        #     x0 = final_search(self.winner_order, self.dir, chessboard, turn, int(100), True)
        #     return x0[0], ((x0[1], x0[2]), )
        if deep <= 0:
            return self.evaluation(chessboard), ()
        if timeout > self.single_time_out:
            return time_shoot, ()
        start_time = time.time()
        any_pos_flag = 0
        final_list = ()
        for ite in self.order_x:
            if get_accessible(chessboard, turn, ite, self.dir):
                any_pos_flag += 1
                # movement
                changed = next_board(chessboard, ite, turn, self.dir)
                nxt = self.ab(chessboard, -turn, deep - 1, alpha, beta, True,
                              timeout + time.time() - start_time)
                chessboard[ite[0]][ite[1]] = COLOR_NONE
                for _ in range(len(changed)):
                    xx, yy = changed[_][0], changed[_][1]
                    chessboard[xx][yy] = -turn
                if nxt[0] > 1234560:
                    return time_shoot, final_list
                if turn == self.color:
                    if alpha < nxt[0]:
                        alpha = nxt[0]
                        final_list = (ite,) + nxt[1]
                    if alpha >= beta:
                        if self.maxDepth == deep:
                            break
                        return alpha, final_list
                else:
                    if beta > nxt[0]:
                        beta = nxt[0]
                        final_list = (ite,) + nxt[1]
                    if alpha >= beta:
                        return beta, final_list
        # At the first layer, returns the position
        if deep == self.maxDepth:
            return alpha, final_list
        # There is no location to go down (at this time to ensure that it will not appear in the first layer)
        elif any_pos_flag == 0:
            sc, my = num_chess(chessboard, self.color)
            if sc == 64 or not last_chance:
                return INF if my < sc // 2 else (0 if my == sc // 2 else -INF), ()
            else:
                return self.ab(chessboard, -turn, deep - 1, alpha, beta, False,
                               timeout + time.time() - start_time)
        return (alpha if turn == self.color else beta), final_list

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        self.candidate_list = get_accessible_list_(chessboard, self.color)
        self.current_num = num_chess(chessboard, 0)[0]

        if self.is_first_round == 2:
            kjk_pos = self.kjk.get(bit_board(chessboard))
            self.is_first_round = 1
            if kjk_pos is not None:
                self.candidate_list.append(kjk_pos)
            final_search(self.winner_order, self.dir, chessboard, self.color, 0, True)
            return None
        elif self.is_first_round == 1:
            self.single_time_out = 3.0
            calculate([0, 0, 0, 0, 0, 0, 0, 0], self.dp_array)
            calculate0(0, 0, 0, 0, 0, 0, 0, 0, self.dp_array)
            self.is_first_round = 0
        else:
            self.single_time_out = 4.9

        if len(self.candidate_list) > 0:
            total_time = 0.0
            depth = 1
            best_list = ()
            best_list_set = set()

            while total_time < self.single_time_out:
                self.order_x.clear()
                best_list_set.clear()
                for i in range(len(best_list)):
                    if best_list[i] != (-1, -1):
                        self.order_x.append(best_list[i])
                        best_list_set.add(best_list[i])
                for i in order0:
                    if i not in best_list_set:
                        self.order_x.append(i)
                self.maxDepth = depth
                timer_ = time.time()
                x0 = self.ab(chessboard, self.color, self.maxDepth, -(INF + 1), INF + 1, True, total_time)
                best_list = x0[1]
                total_time += time.time() - timer_
                if len(x0[1]) > 0 and x0[1][0] != (-1, -1):
                    self.candidate_list.append(x0[1][0])
                depth += 1
                if depth > 1:
                    break

            # if self.current_num >= self.final_search_begin:
            #     print("get winner search!")
            #     x1 = final_search(self.winner_order, self.dir, chessboard, self.color, int(100), True)
            #     print(x1[0])
            #     if x1[0] >= 0:
            #         print("get win or draw!")
            #         self.candidate_list.append((x1[1], x1[2]))
            #         print("search time =", total_time, "win step:", (x1[1], x1[2]))

    # print("time =", time.time() - timer)
    # ==============Find new pos========================================
    # Make sure that the position of your decision on the chess board is empty.
    # If not, the system will return error.
    # Add your decision into candidate_list, Records the chessboard
    # You need to add all the positions which are valid
    # candidate_list example: [(3,3),(4,4)]
basicab = AI(8, 0, 100)
# mct = MCT(8, 0, 100)


def playto(black: AI, white: AI) -> int :
    bd = np.zeros((8, 8), dtype=int)
    bd[3][3] = COLOR_WHITE
    bd[3][4] = COLOR_BLACK
    bd[4][3] = COLOR_BLACK
    bd[4][4] = COLOR_WHITE
    black.color=COLOR_BLACK
    white.color=COLOR_WHITE
    # print(len(black.delta_matrix),len(black.delta_matrix[0]))
    for t in range(61) :
        if t % 2 == 0 :
            black.go(bd)
            c = black.candidate_list
            if len(c) > 0 :
                p = c[len(c) - 1]
                next_board(bd, p, COLOR_BLACK, black.dir)
                bd[p[0]][p[1]] = COLOR_BLACK
        else :
            white.go(bd)
            c = white.candidate_list
            if len(c) > 0 :
                p = c[len(c) - 1]
                next_board(bd, c[len(c) - 1], COLOR_WHITE, white.dir)
                bd[p[0]][p[1]] = COLOR_WHITE
    total = 0
    black_num = 0
    for i in range(8) :
        for j in range(8) :
            total += abs(bd[i][j])
            if bd[i][j] == COLOR_BLACK :
                black_num += 1
    if black_num < total / 2 :
        return 1
    elif black_num == total / 2 :
        return 0
    else :
        return -1


def translate(chessboard, loc) :
    x = loc[0]
    y = loc[1]
    chessboard[x][7 - y] = chessboard[7 - x][y] = chessboard[7 - x][7 - y] = chessboard[x][y]

def get_subv(father:int,mother:int,length:int)->int:
    if type(mother)==str:
        print('m')
    here=0
    for i in range(length):
        if random.randint(0,100000)%2==0:
            digit=(father>>i)&1
        else:
            digit=(mother>>i)&1
        here=here|(digit<<i)
        if(random.randint(0,10000)%20==0):
            here=here^(1<<i)
    return here



def product(waiters: list) :
    l = len(waiters)
    for i in range(l) :
        waiter = waiters[i]
        waiter: AI
        mate: AI = waiters[(i + random.randint(0, l - 1)) % l]
        son = AI(8, 0, 100)
        # son.c1 = waiter.c1 if random.randint(0, 100) % 2 == 0 else mate.c1
        # print(waiter.score_matrix[0])
        # print(mate.score_matrix[0])
        # if random.randint(0, 100000) % 12 == 0 :
        #     son.c1 += int(random.randint(-21, 21) / int(total ** 0.25))
        # son.c2 = waiter.c2 if random.randint(0, 100) % 2 == 0 else mate.c2
        # if random.randint(0, 100000) % 12 == 0 :
        #     for j in range(4) :
        #         son.c2[j] += int(random.randint(-21, 21) / int(total ** 0.25))
        # son.c3 = waiter.c3 if random.randint(0, 100) % 2 == 0 else mate.c3
        # if random.randint(0, 100000) % 12 == 0 :
        #     for j in range(4) :
        #         son.c3[j] += int(random.randint(-21, 21) / int(total ** 0.25))

        for row in range(4) :
            for column in range(row + 1) :
                fu = int(waiter.score_matrix[row][column]) + 64
                mu = int(mate.score_matrix[row][column]) + 64
                here = 0
                here=get_subv(fu,mu,7)-64
                son.score_matrix[row][column] = here
                son.score_matrix[column][row] = son.score_matrix[row][column]
                translate(son.score_matrix, [row, column])
                translate(son.score_matrix, [column, row])

                fu = int(waiter.score_matrix1[row][column]) + 64
                mu = int(mate.score_matrix1[row][column]) + 64
                son.score_matrix1[row][column] = get_subv(fu, mu, 7) - 64
                son.score_matrix1[column][row] = son.score_matrix1[row][column]
                translate(son.score_matrix1, [row, column])
                translate(son.score_matrix1, [column, row])

                fu=int(waiter.arb_matrix[row][column])+64
                mu=int(mate.arb_matrix[row][column])+64
                here = get_subv(fu, mu, 7) - 64
                son.arb_matrix[row][column]=here
                son.arb_matrix[column][row]=here
                translate(son.arb_matrix,[row,column])
                translate(son.arb_matrix,[column,row])

                fu=int(waiter.arb_matrix1[row][column])+64
                mu=int(mate.arb_matrix1[row][column])+64
                here = get_subv(fu, mu, 7) - 64
                son.arb_matrix1[row][column]=here
                son.arb_matrix1[column][row]=here
                translate(son.arb_matrix1,[row,column])
                translate(son.arb_matrix1,[column,row])

        son.c2 = [1 for i in range(3)]
        son.c3 = [1 for i in range(3)]
        son.c4 = [1 for i in range(3)]
        for i in range(3):
            fu = waiter.c2[i]
            mu = mate.c2[i]
            son.c2[i] = get_subv(fu, mu, 8)
            fu=waiter.c3[i]+64
            mu=mate.c3[i]+64
            son.c3[i]=get_subv(fu,mu,7)-64
            fu=-waiter.c4[i]
            mu=-mate.c4[i]
            son.c4[i]=-get_subv(fu,mu,7)

        waiters.append(son)


def sub_inheritance(waiters: list, start: int, end: int) -> list :
    # print(playNum)
    # l1=[0 for i in range(15)]
    lis = [[0 for i in range(len(waiters))] for j in range(3)]
    l=len(waiters)
    for i in range(start, end) :
        black_player = waiters[i]
        black_player.color = COLOR_BLACK
        for j in range(1, playNum + 1) :
            x: int
            white_player=waiters[(i+j)%l]
            white_player.color = COLOR_WHITE
            res = playto(black_player, white_player)
            if res == 1 :
                lis[0][i] += 1
            elif res == -1 :
                lis[1][(i+j)%l] += 1
            else :
                lis[2][i] += 1
                lis[2][(i+j)%l] += 1
    # print("finish",flush=True)
    return lis

    # print("finish", flush=True)


def sub_inheritance_muti(li) :
    return sub_inheritance(li[0], li[1], li[2])

def play_history(waiters: list,start: int, end: int) -> list :
    # print(playNum)
    # l1=[0 for i in range(15)]
    lis = [[0 for i in range(len(waiters))] for j in range(3)]
    l=len(waiters)
    for i in range(start, end) :
        black_player = waiters[i]
        black_player.color = COLOR_BLACK
        for j in range(1,l) :
            x: int
            white_player=waiters[(i+j)%l]
            white_player.color = COLOR_WHITE
            res = playto(black_player, white_player)
            if res == 1 :
                lis[0][i] += 1
            elif res == -1 :
                lis[1][(i+j)%l] += 1
            else :
                lis[2][i] += 1
                lis[2][(i+j)%l] += 1
    # print("finish",flush=True)
    return lis

    # print("finish", flush=True)


def play_history_multi(li) :
    return play_history(li[0], li[1], li[2])

def now_to_history(waiters: list,history:list,start: int, end: int) -> list :
    # print(playNum)
    # l1=[0 for i in range(15)]
    lis = [[0 for i in range(len(waiters))] for j in range(3)]
    l=len(history)
    for i in range(start, end) :
        black_player = waiters[i]
        black_player.color = COLOR_BLACK
        for j in range(l) :
            x: int
            white_player=history[j]
            white_player.color = COLOR_WHITE
            res = playto(black_player, white_player)
            if res == 1 :
                lis[0][i] += 1
            elif res==0:
                lis[2][i] += 1
        white_player=waiters[i]
        white_player.color=COLOR_WHITE
        for j in range(l) :
            x: int
            black_player=history[j]
            black_player.color = COLOR_BLACK
            res = playto(black_player, white_player)
            if res == -1 :
                lis[1][i] += 1
            elif res==0:
                lis[2][i] += 1
    # print("finish",flush=True)
    return lis

    # print("finish", flush=True)



def now_to_history_muti(li):
    return now_to_history(li[0], li[1], li[2],li[3])


history_winner=[]
history_weight=1
def inheritance(waiters: list) :

    # file1 = open("cores_control.txt", mode="r")
    # s1 = file1.readline()
    # cores = int(s1)
    random.seed()
    random.shuffle(waiters)
    for player in waiters :
        player: AI
        player.win_num = 0
        player.tie_num = 0
        player.black_win = player.white_win = 0
        player.history_black_win=player.history_white_win=0
    length = int(waiter_num / cores)
    args = [[waiters, i * length, (i + 1) * length] for i in range(cores)]

    # res=sub_inheritance_muti(args[0])
    # print(args)
    res=[]
    # file = open("control.txt", mode="r")

    # s = file.readline()
    # if s=="5":
    #     sc_file=open("score.txt",mode="r")
    #     for i in range(8):
    #         s_inner=sc_file.readline()
    #         arr=s_inner.split(',')
    #         for j in range(len(arr)):
    #             basicab.score_matrix[i][j] = int(arr[j])
    #
    #     print(basicab.score_matrix)
    with Pool(cores) as p :
        res = p.map(sub_inheritance_muti, args)
    p.join()
    p.close()
    # print("go", flush=True)
    for idx, lis in enumerate(res) :
        for i in range(waiter_num):
            a = waiters[i]
            a: AI
            a.black_win+=res[idx][0][i]
            a.white_win+=res[idx][1][i]
            a.tie_num+=res[idx][2][i]
    for waiter in waiters:
        waiter:AI
        waiter.win_num=min(waiter.white_win,waiter.black_win)
    # normal_res=
    list.sort(waiters, key=lambda ai : (ai.win_num, ai.tie_num), reverse=True)
    winner:AI=waiters[0]
    history_num=int(6)
    if len(history_winner)<history_num-1:
        cp=AI(8,0,100)
        for i in range(8):
            for j in range(8):
                cp.score_matrix[i][j]=winner.score_matrix[i][j]
                cp.score_matrix1[i][j]=winner.score_matrix1[i][j]
                cp.arb_matrix[i][j]=winner.arb_matrix[i][j]
                cp.arb_matrix1[i][j]=winner.arb_matrix1[i][j]
        cp.c1=winner.c1
        for i in range(3):
            cp.c2[i]=winner.c2[i]
            cp.c3[i]=winner.c3[i]
            cp.c4[i]=winner.c4[i]
        history_winner.append(cp)
    else:
        for history in history_winner:
            history:AI
            history.black_win=history.white_win=history.tie_num=history.history_black_win=history.history_white_win=0

        args = [[waiters, history_winner,i * length, (i + 1) * length] for i in range(cores)]
        with Pool(cores) as p2 :
            res_now_to_history = p2.map(now_to_history_muti, args)
        for idx,lis in enumerate(res_now_to_history):
            for i in range(waiter_num):
                a:AI=waiters[i]
                a.black_win+=res_now_to_history[idx][0][i]*history_weight
                a.white_win+=res_now_to_history[idx][1][i]*history_weight
                a.tie_num+=res_now_to_history[idx][2][i]*history_weight
                a.history_black_win+=res_now_to_history[idx][0][i]*history_weight
                a.history_white_win+=res_now_to_history[idx][1][i]*history_weight
        for wt in waiters:
            wt.win_num=min(wt.black_win,wt.white_win)
        list.sort(waiters, key=lambda ai : (ai.win_num, ai.tie_num), reverse=True)
        winner: AI = waiters[0]

        cp = AI(8, 0, 100)
        for i in range(8) :
            for j in range(8) :
                cp.score_matrix[i][j] = winner.score_matrix[i][j]
                cp.score_matrix1[i][j] = winner.score_matrix1[i][j]
                cp.arb_matrix[i][j]=winner.arb_matrix[i][j]
                cp.arb_matrix1[i][j]=winner.arb_matrix1[i][j]
        cp.c1 = winner.c1
        cp.live_time=total
        for i in range(3) :
            cp.c2[i] = winner.c2[i]
            cp.c3[i] = winner.c3[i]
            cp.c4[i] = winner.c4[i]
        history_winner.append(cp)

        length = 1
        args=[[history_winner,i*length,(i+1)*length] for i in range(history_num)]
        for wt in history_winner:
            wt.win_num=0
            wt.black_win=0
            wt.white_win=0
            wt.tie_num=0
        with Pool(history_num) as p1 :
            res_history = p1.map(play_history_multi, args)
        p1.join()
        p1.close()
        for idx,lis in enumerate(res_history):
            for i in range(history_num):
                a:AI=history_winner[i]
                a.black_win+=res_history[idx][0][i]
                a.white_win+=res_history[idx][1][i]
                a.tie_num+=res_history[idx][2][i]
        for wt in history_winner:
            wt.win_num=min(wt.black_win,wt.white_win)
        list.sort(history_winner,key=lambda ai:(ai.win_num,ai.tie_num),reverse=True)
        history_winner.pop()
    history_out = open("history.txt", mode="w")
    for waiter in waiters:
        waiter:AI
        waiter.win_num=min(waiter.white_win,waiter.black_win)
    for wt in history_winner:
        wt:AI
        history_out.writelines(['life time:' + str(wt.live_time), ',black_wins:' + str(wt.black_win) + ',white_wins:' + str(
            wt.white_win) + ',total_win:' + str(wt.win_num)  +',history_black_win:'+str(wt.history_black_win) +',history_white_win:'+str(wt.history_white_win)+'ties:'+ str(wt.tie_num) + "\n",
                        str(wt.score_matrix) + "\n", str(wt.score_matrix1) + "\n",str(wt.arb_matrix)+"\n",str(wt.arb_matrix1)+"\n",
                        str((str(wt.c1), str(wt.c2), str(wt.c3), str(wt.c4))) + "\n"])
    history_out.flush()
    history_out.close()
    if True:
        out = open("log.txt", mode="w")
        for i in range(int(waiter_num/2)) :
            wt = waiters[i]
            wt:AI
            out.writelines(['life time:'+str(wt.live_time),',black_wins:' + str(wt.black_win) + ',white_wins:' + str(
                wt.white_win) + ',total_win:' + str(wt.win_num)  +',history_black_win:'+str(wt.history_black_win) +',history_white_win:'+str(wt.history_white_win)+ ',ties:' + str(wt.tie_num) + "\n",
                            str(wt.score_matrix.tolist()) + "\n",str(wt.score_matrix1.tolist())+"\n", str(wt.arb_matrix.tolist())+"\n",str(wt.arb_matrix1.tolist())+"\n",str((str(wt.c1), str(wt.c2), str(wt.c3),str(wt.c4)))+"\n"])
        out.flush()
        out.close()
    # file.close()


    for t in range(int(waiter_num / 2)) :
        waiters.pop()
        waiters[t].live_time+=1
    list.sort(waiters, key=lambda ai : (ai.live_time,ai.win_num, ai.tie_num), reverse=True)
    winner: AI = waiters[0]
    print(f"life time:{winner.live_time},black_win:{winner.black_win},white_win:{winner.white_win},total_win:{winner.win_num},ties:{winner.tie_num},history_black_win:{winner.history_black_win},history_white_win:{winner.history_white_win}")
    print(f"{winner.score_matrix}")
    print(f"{winner.score_matrix1}")
    print(f"{winner.arb_matrix}")
    print(f"{winner.arb_matrix1}")
    print((winner.c1, winner.c2, winner.c3,winner.c4))
    product(waiters)
    # for i in range(int(waiter_num/2)):


total = 1
cores = 8

#
# def read(s:str)->int:
#     for ch in s:
#         if str.isdigit(ch):


if __name__ == "__main__" :
    random.seed()
    # playNum=4
    # waiter_num=180
    basicab.score_matrix = [[1 for i in range(8)] for j in range(8)]
    basicab.score_matrix1 = [[1 for i in range(8)] for j in range(8)]
    basicab.delta_matrix = [[0 for i in range(8)] for j in range(8)]
    basicab.c1 = 1
    basicab.c2 = [0 for i in range(3)]
    basicab.c3 = [0 for i in range(3)]
    basicab.c4 = [0 for i in range(3)]
    waiters = []
    f = open("log.txt")
    for i in range(int(waiter_num // 2)):
        here = AI(8, 0, 100)
        waiters.append(here)
        f.readline()
        s = f.readline()
        digits = re.findall(r'-?\d+', s)
        for ii in range(8):
            for jj in range(8):
                here.score_matrix[ii][jj] = int(digits[ii * 8 + jj])
        s = f.readline()
        digits = re.findall(r'-?\d+', s)
        for ii in range(8):
            digits = re.findall(r'-?\d+', s)
            for jj in range(8):
                here.score_matrix1[ii][jj] = int(digits[ii * 8 + jj])
        s = f.readline()
        digits = re.findall(r'-?\d+', s)
        for ii in range(8):
            digits = re.findall(r'-?\d+', s)
            for jj in range(8):
                here.arb_matrix[ii][jj] = int(digits[ii * 8 + jj])
        s = f.readline()
        digits = re.findall(r'-?\d+', s)
        for ii in range(8):
            digits = re.findall(r'-?\d+', s)
            for jj in range(8):
                here.arb_matrix1[ii][jj] = int(digits[ii * 8 + jj])
        s = f.readline()
        digits = re.findall(r'-?\d+', s)
        here.c1 = 1
        for ii in range(3):
            here.c2[ii] = int(digits[ii + 1])
        for ii in range(3):
            here.c3[ii] = int(digits[ii + 4])
        for ii in range(3):
            here.c4[ii] = int(digits[ii + 7])
        # for ii in range(3):
        #     here.c5[ii] = int(digits[ii + 10])
    product(waiters)

    #     for j in range(3) :
    #         here.c2[j] = 0
    #         here.c3[j] = random.randint(-64, 63)
    #         here.c4[j] = random.randint(-127, 0)
    #         here.c5[j] = 0
    #     for row in range(4) :
    #         for column in range(row + 1) :
    #             v = random.randint(-64, 63)
    #             v1 = random.randint(-64, 63)
    #             here.score_matrix[row][column] = v
    #             here.score_matrix[column][row] = here.score_matrix[row][column]
    #             here.score_matrix1[row][column] = v1
    #             here.score_matrix1[column][row] = here.score_matrix1[row][column]
    #             v = random.randint(-64, 63)
    #             v1 = random.randint(-64, 63)
    #             here.arb_matrix[row][column] = v
    #             here.arb_matrix[column][row] = v
    #             here.arb_matrix1[row][column] = v1
    #             here.arb_matrix1[column][row] = v1
    #
    #             translate(here.score_matrix, [row, column])
    #             translate(here.score_matrix, [column, row])
    #             translate(here.score_matrix1, [row, column])
    #             translate(here.score_matrix1, [column, row])
    #             translate(here.arb_matrix, [row, column])
    #             translate(here.arb_matrix, [column, row])
    #             translate(here.arb_matrix1, [row, column])
    #             translate(here.arb_matrix1, [column, row])
    #             # translate(here.delta_matrix, [row, column])
    #             # translate(here.delta_matrix, [column, row])
    #
    #     here.c1 = 1
    #     # here.c2 = [40 + random.randint(-30, 30) for i1 in range(4)]
    #     # here.c3 = [5 + random.randint(-5, 50) for i1 in range(4)]
    # for i in range(5) :
    #     here = waiters[i]
    #     v = random.randint(-64, 63)
    #     v1 = random.randint(-64, 63)
    #     for row in range(4) :
    #         for column in range(row + 1) :
    #             here.score_matrix[row][column] = v
    #             here.score_matrix[column][row] = here.score_matrix[row][column]
    #             here.score_matrix1[row][column] = v1
    #             here.score_matrix1[column][row] = here.score_matrix1[row][column]
    #             here.arb_matrix[row][column] = 0
    #             here.arb_matrix[column][row] = 0
    #             here.arb_matrix1[row][column] = 0
    #             here.arb_matrix1[column][row] = 0
    #             # here.delta_matrix[row][column] = random.randint(-100, 100)
    #             # here.delta_matrix[column][row] = here.delta_matrix[row][column]
    #             translate(here.score_matrix, [row, column])
    #             translate(here.score_matrix, [column, row])
    #             translate(here.score_matrix1, [row, column])
    #             translate(here.score_matrix1, [column, row])
    #             translate(here.arb_matrix, [row, column])
    #             translate(here.arb_matrix, [column, row])
    #             translate(here.arb_matrix1, [row, column])
    #             translate(here.arb_matrix1, [column, row])
    #             # translate(here.delta_matrix, [row, column])
    #             # translate(here.delta_matrix, [column, row])
    #     for j in range(3) :
    #         here.c3[j] = 0
    #         here.c4[j] = 0
    #         here.c5[j] = 0
    print("starting...")
    while True :
        random.seed()
        inheritance(waiters)
        print(f"total:{total}", flush=True)
        total += 1