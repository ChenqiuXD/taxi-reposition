import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.plot(np.arange(1000), [1/(i+1) for i in range(1000)])
plt.show()