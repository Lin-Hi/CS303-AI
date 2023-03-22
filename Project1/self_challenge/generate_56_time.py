from Project1 import phase6, phase5
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
    f = open('5_6_time.txt','a')
    try:
        while True:
            AI = phase6.AI(8, COLOR_WHITE, 5)
            t1 = time.time()
            AI.go(chessboard)
            ph6_time = time.time() - t1
            AI = phase5.AI(8, COLOR_WHITE, 5)
            t1 = time.time()
            AI.go(chessboard)
            ph5_time = time.time() - t1
            print(f"ph5_time={ph5_time}, ph6_time={ph6_time}")
            f.write(f"ph5_time={ph5_time}, ph6_time={ph6_time}\n")
    except BaseException:
        f.flush()
    finally:
        f.close()
