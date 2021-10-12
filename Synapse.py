from collections import deque

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

syn = Synapse(3,2, d=10)
print(syn.que)