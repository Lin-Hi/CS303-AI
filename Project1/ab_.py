import time
import numpy as np


def ab(self, chessboard: np.array, turn, deep, alpha, beta, last_chance, timeout) -> tuple:
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