from collections import deque
import numpy as np
import time
import random

class Synapse:
    def __init__(self, i, j, w=1, d=1):
        self.i = i
        self.j = j
        self.w = w
        self.d = d
        self.que = deque(False for x in range(d))

    def pop(self):
        return self.que.pop()

    def push(self, spike):
        self.que.appendleft(spike)

    def update(self, change):
        for i in range(abs(change)):
            if change > 0:
                self.que.appendleft(False)
            elif change < 0:
                self.que.popleft()

syn = Synapse(3,5)

list1 = [syn]
list2 = [syn]

list1[0].i = 23
print(list2[0].i)
list2[0].i = 100
print(list1[0].i)

'''
start = time.time()
N = 1000
dict_of_dict = {}
for i in range(N):
    dict_of_dict.setdefault(i, {})
    for j in range(N):
        dict_of_dict[i][j] = Synapse(i,j)
end = time.time()
print(end-start)
print(dict_of_dict[)




start = time.time()
N = 1000
list = [[] for _ in range(N)]
for i in range(N):
    for j in range(N):
        list[i].append(Synapse(i,j))
for i in range(100000):
    x = list[random.randint(0,999)][random.randint(0,999)]
end = time.time()
print(end-start)
print(list[2][:])
'''