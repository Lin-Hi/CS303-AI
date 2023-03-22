from Project1 import phase6, phase5, phase5_noAB
import time
import numpy as np
import random

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(time.time())
TIME_BEFORE_OUT = 0.15

chessboard = np.asarray([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, -1, 1, -1, -1, 0, 0],
    [0, 0, 1, 1, 1, -1, 0, 0],
    [0, 0, 0, -1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

if __name__ == '__main__':
    f = open('ab.txt', 'a')
    try:
        for _ in range(10_0000):
            AI = phase5.AI(8, COLOR_WHITE, 5)
            t1 = time.time()
            AI.go(chessboard)
            ph5_time = time.time() - t1
            AI = phase5_noAB.AI(8, COLOR_WHITE, 5)
            t1 = time.time()
            AI.go(chessboard)
            ph5_noAB = time.time() - t1
            print(f"ph5 with ab = {ph5_time}, ph5 without ab = {ph5_noAB}")
            f.write(f"ph5 with ab = {ph5_time}, ph5 without ab = {ph5_noAB}\n")
    except BaseException:
        f.flush()
    finally:
        f.flush()
        f.close()
