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
# print(my_matrix[[0, 3, 4], 1:])


# a_mul = np.array([
#     [
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#     ],
#     [
#         [5, 6, 7, 8],
#         [9, 10, 11, 12],
#         [13, 14, 15, 16],
#      ],
# ])
#
# print(a_mul.shape)
# print(a_mul.size)
# print(a_mul.ndim)

# my_dict = {
#     'one': 1,
#     'two': 2,
#     'three': 3,
# }
#
# arr = np.array([
#     [
#         [1, 2, 3, 4],
#         [1, 2, 3, 4]
#     ],
#     [
#         [5, 6, my_dict, 8],
#         [9, 10, 11, 12],
#     ]
# ])
#
#
# print(arr.dtype)
# print(type(arr[0, 0, 0]))
# print(type(arr[1, 0, 2]))

# a = np.ones((3,5), dtype='int64')
# print(a)

# b = np.full((3, 2, 5), 69, dtype='int64')
# print(b)

# c = np.arange(0, 1005, 5)
# print(c)
#
# d = np.arange(0, 1005)
# print(d)

"""
NumPy Array vs. Python list
"""

# l1 = [1, 2, 3, 4, 5]
# l2 = [6, 7, 8, 9, 10]
#
# a1 = np.array(l1)
# a2 = np.array(l2)

# print(l1 * 5)
# print(a1 * 5)

# print(l1 + l2)
# print(a1 + a2)

# print(a1 / a2)
# print(a1 - a2)

"""
Array methods
"""

# a = np.array(
#     [[1, 2],
#      [3, 4]]
# )
# print(a)
# print(np.append(a, [69, 68]))


# print(np.delete(a, 3))
# print(np.delete(a, 0, axis=0))
# print(np.delete(a, 1, axis=0))


# print(np.delete(a, 0, axis=1))
# print(np.delete(a, 1, axis=1))


"""
Structuring methods
"""

# a = np.array(
#     [[1, 2, 3, 4, 5],
#      [6, 7, 8, 9, 10],
#      [11, 12, 13, 14, 15],
#      [16, 17, 18, 19, 20]]
# )

# print(np.shape(a))
# print(np.reshape(a, (10, 2)))
# print(np.reshape(a, (2, 10)))
# new_a = np.reshape(a, (2, 5, 2))
# print(new_a)
# print(np.ndim(new_a))

# print(a.reshape((2, 5, 2)))
# print(a.reshape((2, 10)))

# print(np.reshape(a, (20, )))
# print(np.reshape(a, (20, 1)))

# print(a.flatten())
# print(a)

# print(a.transpose())

"""
Concatenate arrays
"""

# a1 = np.array(
#     [[1, 2, 3, 4, 5],
#      [6, 7, 8, 9, 10]]
# )
#
# a2 = np.array(
#     [[11, 12, 13, 14, 15],
#      [16, 17, 18, 19, 20]]
# )

# print(np.concatenate((a1, a2), axis=0))
# print(np.concatenate((a1, a2), axis=1))


"""
Stacking arrays -> Adding dimension
"""

# new_a = np.stack((a1, a2))
# print(new_a)
# print(new_a.ndim)


"""
Split array
"""

# a = np.array(
#     [[1, 2, 3, 4, 5, 6],
#      [7, 8, 9, 10, 11, 12],
#      [15, 16, 17, 18, 19, 20],
#      [21, 22, 23, 24, 25, 26],
#      [27, 28, 29, 30, 31, 32],
#      [33, 34, 35, 36, 37, 38]]
# )

# print(np.split(a, 2, axis=0))
# print(np.split(a, 3, axis=0))
# print(np.split(a, 6, axis=0))

# print(np.split(a, 3, axis=1))
# print(np.split(a, 6, axis=1))
# print(np.split(a, 2, axis=1))


"""
NumPy random
"""

# a = np.random.randint(100, size=(3, 3))
# print(a)

# b = np.random.randint(100, size=(3, 3, 5))
# print(b)

# c = np.random.randint(5, 10, (3, 5, 6))
# print(c)


"""
Importing and Exporting array
"""

# a = np.array(
#     [[1, 2],
#      [3, 4],
#      [5, 6]]
# )

# np.savetxt("my_matrix.txt", a, delimiter=',')

# a = np.loadtxt('my_matrix.txt', delimiter=',').astype(int)
# print(a)
#
# a = np.genfromtxt('my_matrix.txt', delimiter=',').astype(int)
# print(a)

"""
Exercise
"""

# a_mul = np.array([
#     [
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#     ],
#     [
#         [5, 6, 7, 8],
#         [9, 10, 11, 12],
#         [13, 14, 15, 16],
#      ],
# ])


# a_mul[:, 1:, 2:] = 9
# print(a_mul)


"""
Sorting array
"""

# a_mul = np.array([
#     [
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#     ],
#     [
#         [5, 6, 7, 8],
#         [89, 10, 11, 12],
#         [69, 14, 15, 16],
#      ],
# ])
#
#
# a = np.sort(a_mul, axis=-1)
# b = np.sort(a_mul, axis=0)
# print(a)

"""
Sum of columns and rows
"""

# a = np.full((3, 4), 9)
# a += 1
# print(a)
# print(np.sum(a, axis=1))
# print(np.sum(a, axis=0))

# a = np.array(
#     [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# )

# print(a.sum())
# print(np.prod(a))
# print(a.prod())
# print(np.mean(a))
# print(np.max(a))
# print(np.min(a))
# print(a.min())
# print(a.max())

"""
peak_to_peak -> finding the value of max - min
"""

# print(np.ptp(a))

"""
Finding the index of max and min value of matrix
"""

# print(np.argmax(a))
# print(np.argmin(a))
# print(a.argmin())
# print(a.argmin())

# a = np.array(
#     [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# )
#
# print(a[[0, 0, 2], [0, 1, 2]])
# print(a[[0, 1, 2], [0, 1, 2]])
