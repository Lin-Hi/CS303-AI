import heapq

if __name__ == '__main__':
    nums = [1,3,5,4,2]
    h = []
    heapq.heapify(nums)
    l = len(nums)
    for _ in range(l):
        print(heapq.heappop(nums))