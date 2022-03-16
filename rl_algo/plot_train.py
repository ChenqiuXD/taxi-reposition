import matplotlib.pyplot as plt
import numpy as np

result_iql = np.load("iql.npy")
result_vdn = np.load("vdn.npy")

length = 1000
plt.subplot(121)
plt.plot(range(length), result_iql[:length])
plt.title("iql methods")
plt.subplot(122)
plt.plot(range(length), result_vdn[:length])
plt.title("vdn methods")

# tmp = np.array([1,2,3,4,5,6,7,8,9])
# plt.plot(range(tmp.shape[0]), tmp)

plt.show()
