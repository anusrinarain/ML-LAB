import numpy as np
arr = np.arange(1, 13)
print("Original Array:")
print(arr)
print("Shape:", arr.shape)
reshaped_arr = arr.reshape(3,4)
print("\nReshaped Array (3x4):")
print(reshaped_arr)
print("Shape:", reshaped_arr.shape)