import phase2
import random
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    total_times = 0
    y_my_num = []
    y_ran_num = []
    y_equ_num = []
    y_my_percent = []
    y_ran_percent = []
    y_equ_percent = []
    with open('self_challenge/ran_2_result.txt', 'r') as f:
        for line in f.readlines():
            arr1 = line.split(': ')
            times = int(arr1[0])
            total_times = times
            arr2 = arr1[1].split(', ')
            my_win = int(arr2[0].split('=')[1])
            ran_win = int(arr2[1].split('=')[1])
            equ = int(arr2[2].split('=')[1])
            y_my_num.append(my_win)
            y_my_percent.append(my_win / times)
            y_ran_num.append(ran_win)
            y_ran_percent.append(ran_win / times)
            y_equ_num.append(equ)
            y_equ_percent.append(equ / times)

    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    x_axis_data = [i * 10 for i in range(1, times//10+1)]
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y_equ_percent,label="dogfall")
    plt.plot(x_axis_data, y_my_percent,label="alpha-beta")
    plt.plot(x_axis_data, y_ran_percent,label="random")
    # plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='一些数字')
    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="best")
    plt.xlabel('Simulation times')
    plt.ylabel('Winning times')
    plt.show()
    # print("1: 22".split(": "))
