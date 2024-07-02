import numpy as np

x = np.array([-1.0, 1.0, 2.0])
y = x > 0
y = y.astype(np.int8)

print(y)
