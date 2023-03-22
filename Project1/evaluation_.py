COLOR_NONE = 0
INF = float('inf')
Dir = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def evaluation(self, chessboard: list):
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
    sc, my = self.num_chess(chessboard, self.color)
    # sc: total chess number, my: this color number
    arb_my = self.get_accessible_list(chessboard, self.color)
    # 我可以下子的地方
    arb_opp = self.get_accessible_list(chessboard, -self.color)
    # 对方可以下子的地方

    if len(arb_my) == 0 and len(arb_opp) == 0:
        # 判断终局
        return INF if my < sc // 2 else (0 if my == sc // 2 else -INF)

    stable_my, stable_opp = self.stable(chessboard, is_stable)
    # 双方的稳定子
    stable = (stable_opp - stable_my) * (1 + 0.2 * max(stable_opp, stable_my))
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
                initial += self.score_matrix[i][j] if chessboard[i][j] != self.color else -self.score_matrix[i][j]
            else:
                initial += self.score_matrix1[i][j] if chessboard[i][j] != self.color else -self.score_matrix1[i][j]

    # 算行动力，也用 8x8 数组
    for action in arb_my:
        arb += self.arb_matrix[action[0]][action[1]] if sc <= 44 else self.arb_matrix1[action[0]][action[1]]
    for action in arb_opp:
        arb -= self.arb_matrix[action[0]][action[1]] if sc <= 44 else self.arb_matrix1[action[0]][action[1]]

    for i in range(self.chessboard_size):
        for j in range(self.chessboard_size):
            # 暴力找边缘子
            if chessboard[i][j] != COLOR_NONE and not is_stable[i][j]:
                # Dir: 八个方向的 list
                for kk in Dir:
                    a = i + kk[0]
                    b = j + kk[1]
                    if self.legal((a, b)) and chessboard[a][b] == COLOR_NONE:
                        front += 1 if chessboard[i][j] == self.color else -1

    result = self.c1 * initial + self.c2[k] * stable + self.c3[k] * diff + self.c4[k] * front + arb
    return result
