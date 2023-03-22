if __name__ == '__main__':
    f1 = open('5_7_result.txt', 'r')
    f2 = open('5.txt', 'r')
    f3 = open('7.txt', 'r')
    i = 0
    t5 = 0.0
    c5 = 0
    t7 = 0.0
    c7 = 0
    i5, i7 = 0, 0
    for line in f2.readlines():
        if len(line.split(' ')) == 2:
            tt5 = float(line.split(' ')[1])
            t5 += tt5
            c5 += 1 if tt5 > 5 else 0
            i5 += 1
    for line in f3.readlines():
        if len(line.split(' ')) == 2:
            tt7 = float(line.split(' ')[1])
            t7 += tt7
            c7 += 1 if tt7 > 5 else 0
            i7 += 1
    print(t5/i5)
    print(c5)
    print(t7/i7)
    print(c7)