import numpy as np
def create_grid_connections(synapse, dim):
    x = dim[0]
    y = dim[1]
    pairs = []
    mat = np.arange(x*y).reshape(x,y)
    i = 0
    for row in range(x):
        for col in range(y):
            for x in range(-1,2):
                for y in range(-1,2):
                    x = x + row
                    y = y + col
                    if (row != x or col != y) and x >= 0 and y >= 0:
                        try:
                            pairs.append((mat[row][col],mat[x][y]))
                        except:
                            pass
    return pairs


pairs = create_grid_connections("", (10,15))

print(pairs)