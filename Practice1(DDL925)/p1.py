def two_sum(sequence, t):
    """
    Args:
        sequence: the given sequence as a list
        t: the given target number, which should be the sum of two selected integers.

    Returns:
        res: A list of tuple. And each tuple would be the idx of two selected integers.
    Example:
        input:
        1 2 3 4
        5
        output:
        0 3
        1 2

    """

    appear_dic = {}
    res = []

    for i in range(len(sequence)):
        num = int(sequence[i])
        if not num in appear_dic:
            appear_dic[num] = [i]
        else:
            appear_dic[num].append(i)

    diff_seq = list(appear_dic.keys())
    diff_seq.sort()
    start = 0
    end = len(diff_seq) - 1

    while start != end:
        start_num = diff_seq[start]
        end_num = diff_seq[end]
        if start_num + end_num == t:
            start_list = appear_dic[start_num]
            end_list = appear_dic[end_num]
            for s in start_list:
                for e in end_list:
                    res.append((s, e))
            end -= 1
        elif start_num + end_num > t:
            end -= 1
        else:
            start += 1

    return res


def broot_force(sequence, t):
    res = []
    for x in range(len(sequence)):
        for y in range(x + 1, len(sequence)):
            if x == y:
                continue
            if sequence[x] + sequence[y] == t:
                res.append((x, y))
    return res


# test block
if __name__ == '__main__':
    for i in range(3):
        print(f'case {i}')
        with open(f'./test_cases/problem1/{i + 1}.txt', 'r') as f:
            seq, tar = f.read().strip().split('\n')
            seq = [*map(int, seq.split(' '))]
            tar = int(tar)

        for item in broot_force(seq, tar):
            print('%d %d' % item)
