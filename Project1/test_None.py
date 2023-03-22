import numpy as np
import time
import random
import phase4

COLOR_NONE = 0


def random_board():
    random.seed(time.time())
    chessboard = np.zeros((8, 8), dtype='int')
    for i in range(8):
        for j in range(8):
            k = random.randint(1, 3)
            if k == 1:
                chessboard[i][j] = 0
            elif k==2:
                chessboard[i][j] = 1
            else:
                chessboard[i][j] = -1
    return chessboard


if __name__ == '__main__':
    chessboard = np.array([
        [ 1, 1,  1,  1,  1,  1,  1,  1],
        [-1, 1,  1,  1, -1, -1, -1,  1],
        [-1, 1,  1,  1,  1,  1, -1,  1],
        [-1,-1, -1,  1,  1,  1, -1,  1],
        [-1,-1,  1, -1, -1,  1,  1,  1],
        [-1,-1, -1, -1, -1,  1,  1,  1],
        [-1, 1,  1,  1,  1,  1,  1,  1],
        [ 1, 1,  1,  1, -1, -1,  0, -1]
    ])
    for i in range(1):
        # chessboard = random_board()
        AI = phase4.AI(8,-1,5)
        step = AI.go(chessboard)
        if step is None:
            print(chessboard)
        if (i + 1) % 1 == 0:
            print(i + 1)
    print('done')
