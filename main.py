import numpy as np

# a = np.array([[1, 2, 3, 4, 78],
#               [5, 6, 7, 8, 46],
#               [5, 6, 7, 6, 12],
#               [7, 8, 9, 44, 31]])
#
# arr_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# arr_2 = np.array([5, 4, 12, 3, 8, 89, 7, 1, 9])

# print(type(a))
# print(a.size)
# print(a.shape)
# print(a.ndim)
# print(arr_1 * arr_2)
# print(arr_1[5:8])
# print(a[2, 2])
# print(arr_1)
# print(a[1])
# print(a[0, 1:])
# print(a[:, 1])
# print(arr_1[1:-1:2])
# print(a[0, 1:-1:2])
# a[0, 2] = 33
# print(a)
# b = np.array([[[1, 2, 3, 4, 78],
#                [5, 6, 7, 8, 46]],
#               [[5, 6, 7, 6, 12],
#                [7, 8, 9, 44, 31]]])

# print(b[:, 1, :])
# c = np.zeros((2, 3, 3, 4))
# print(c[0, 2, 2, 3])

# d = np.ones((2, 3), dtype=int)
# print(d)

# e = np.full((2, 3), 5, dtype=float)
# print(e)


# rng = np.random.rand(4, 3, 5)
# print(rng)


# square_matrix = np.identity(8)
# print(square_matrix)


# f = np.array([[1, 2, 3, 4, 5]])
# print(np.repeat(f, 3, axis=1))


# result = np.ones((5, 5), dtype=int)
# result[1:4, 1:4] = 0
# result[2, 2] = 9
# print(result)


# stats = np.array([[1, 2, 3, 4, 5],
#                   [6, 7, 8, 9, 10]])

# print(stats)
# print(stats.min())
# print(stats.mean())
# print(stats.max())
# print(stats.std())
# print(np.max(stats))
# print(np.max(stats, axis=1))
# print(np.max(stats, axis=0))

# before = np.array([[1, 2, 3, 4],
#                    [6, 7, 8, 9]])
#
# after = before.reshape((2, 2, 2))
# print(after)

# before_stats = np.array([[1, 2, 3, 4, 5],
#                   [6, 7, 8, 9, 10]])
#
# after_stats = before_stats.reshape((5, 2, 1))
# print(after_stats)

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])

# vertical_result = np.vstack((a, b, a, a, b))
# print(vertical_result)

# horizontal_result = np.hstack((a, b))
# print(horizontal_result)

# data = np.genfromtxt('data.txt', delimiter=',', dtype=int)
# print(data)
# print(data <= 1)
# print(data[data >= 100])

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(a)
# print(np.unique(a))
# print(a[[5, -2, -1]])

# a = np.any(data > 100, axis=0)
# b = np.any(data > 100, axis=1)
# print(data)
# print(a)
# print(b)

"""
Exercise
"""

# my_matrix = np.array(
#     [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9],
#      [10, 11, 12],
#      [13, 14, 15]]
# )

"""
Find 1, 5, 9 in 'my_matrix'
"""
# print(my_matrix[[0, 1, 2], [0, 1, 2]])


"""
Find 7, 8, 12 in 'my_matrix'
"""
# print(np.array([my_matrix[2, 0], my_matrix[2, 1], my_matrix[3, 2]]))

"""
Find 2, 3, 11, 12, 15, 15 in 'my_matrix'
"""
#
# print(my_matrix[[0, 3, 4], 1:])












