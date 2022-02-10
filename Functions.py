from scipy import stats
import numpy as np
from collections import Counter
l = np.random.poisson(lam=2, size=100)

print(Counter(l))
print(l)
