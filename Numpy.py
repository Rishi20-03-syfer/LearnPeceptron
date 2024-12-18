import numpy as np

# Data types in numpy
# i - integer
# b - boolean
# u - unsigned integer
# f - float
# c - complex float
# m - timedelta
# M - datetime
# O - object
# S - string
# U - unicode string
# V - fixed chunk of memory for other type ( void )
# arr = np.array([1,2,3,4],dtype='S') # Converting interger to string type array
# print(arr)


# creating numpy arrays
# 1st and eaisest way..
# li = [1,2,3,4]
# array = np.array(li)
# print(type(array))

#2nd way by specifing data type of array
# li = [1,2,3,4,45]
# array = np.array(li,dtype=np.float32)
# print(array)
# print(type(array))

#3rd and most use way building array using functions
# ones = np.ones(10)
# print(ones)
# zeros = np.zeros(10)
# print(zeros)
# range_num = np.arange(0,10,2)
# print(range_num)
# line_sperated = np.linspace(0,1,3)  # linespace(start,stop,step)
# print(line_sperated)
# eye_array = np.eye(3,4)  # gives identity matix of given row and column.
# print(eye_array)
# higher dimensional array
# arr = np.array([1,2,3,4,5,6],ndmin=5)
# print(arr)

# Random floats
# arr = np.random.rand(10) # 10 float values between 0 to 1(excluded).
# print(arr) 
# arr = np.random.rand(2,3) # 2 rows 3 columns.
# print(arr)
# arr = np.random.rand(2,3,4) # 3D array or tensor.
# print(arr)
# Random Integers
# arr = np.random.randint(10)
# print(arr)   # gives any single random integer from 0 to 10
# arr = np.random.randint(10,size = 5)
# print(arr)    # 1d array of any 5 integers from 0 to 10
# arr = np.random.randint(10,size=(2,3,4))
# print(arr)      # 3d array repitative 10 intergers similary you can also build 2d array as well.
# Random choice
# arr = np.array([1,2,3,4,5])
# print(np.random.choice(arr))

# Array attributes

# arr= np.array([[1,2,3],[4,5,6]])
# print(arr.ndim)
# print(arr.shape)
# print(arr.size)
# print(arr.dtype)
# print(arr.itemsize)

# copy and view of array
# copy
# arr = np.array([1,2,3])
# x = arr.copy()
# arr[0] = 11
# x[0] = 12
# print(arr,x)   #Deep copy i.e changes made to arr will not be affected to x
# view
# arr = np.array([1,2,3,4])
# x = arr.view()
# print("No changes made ",arr,x)
# arr[0] = 11
# print("changes made to array ",arr,x)  # kind of shallow copy i.e changes made to either x or array reflects to x or array.
# x[0] = 12
# print("changes made to x ",arr,x)


#indexing and slicing
# one_d_array = np.array([1,2,3,4,5])
# Here slicing is same as normal list.
# n_d_array = np.array([[11,2,3],[6,3,7],[35,72,98]])
# print(n_d_array[0,1])  # at row 0 col 1 i.e 2
# print(n_d_array[1,:])  # entire row 1 
# print(n_d_array[:,2])  # every row but col 2

# opration on array
# Every operation works element wise hence every array has to be same size.
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print(a+b) 
# print(a-b)
# print(a*b)
# print(a/b)
# print(a**2)
# print(np.dot(a,b))  # 1*4 + 2*5 + 3*6 = 4+10+18 = 32

# a = np.array([[1,2,3],[4,5,6],[23,54,72]])  # shape of both matirx should be same and square
# b = np.array([[7,8,9],[11,12,13],[12,44,56]])
# print(np.dot(a,b))

# reshaping array
# arr = np.arange(12)
# new_arr = arr.reshape(3,4)
# print(new_arr) 
# yet_new_array = new_arr.flatten()
# print(yet_new_array)
# print(arr.ravel())
# Difference between ravel() and flatten() is that chagnes made in ravel()-ed array will effect in origianl array as well but flatten() array will not
# affect the original array

# Broadcasting...
# it is concept in which 2 arrays are made compatiable on fly to execute certain opration if they have differet shape or size.
# a = np.array([[1,2,3],[4,5,6]]) # for broadcasting to happen either both have same or one should have 1 dimension
# b = np.array([5])
# print(a + b)
# a = np.array([[1],[2],[3]])  # this is broadcasted to [1,1,1],[2,2,2],[3,3,3]
# b = np.array([4,5,6])        # this broadcasted to [4,5,6],[4,5,6],[4,5,6]
# print(a+b)

# Array transposing
# arr = np.array([[1,2,3],[4,5,6]])
# print(arr.transpose())  # both transpose and T do same thing changing rows to columns.
# print(arr.T)

#Stacking and concatination of arrays
# arr = np.array([[1,2,3],[4,5,6]])
# arr2 = np.array([[7,8,9],[11,12,13]])
# vertical_stack = np.vstack((arr,arr2))  # Note vstack((arr1,arr2)) parameters in bouble rounds brackests # alongs columns
# horizontal_stack = np.hstack((arr,arr2)) # alongs rows
# depth_stack = np.dstack((arr,arr2))      # along depth/height
# print(depth_stack)
# Another way to concatinate two array using axis.
# print(np.concatenate((arr,arr2),axis=1))  # axis 1 means along y-axis same output vstack().
# print(np.concatenate((arr,arr2),axis=0))    # axis 0 means along x-axis same output as hstack().

# Spliting array into simpler array or sub array
# arr = np.arange(16).reshape(4,4)
# print(arr)
# print(np.vsplit(arr,2))
# print(np.hsplit(arr,2))  # Note vsplit and hsplit supports only equal division that means 16 can not be divided into 3 parts it can be 4 instead.
# print(np.array_split(arr,3)) #np.array_split can split the array in any shape reguardless of array shape.

# Linear Algebra on array
# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# # Matrix mulitplication
# matrix_product = np.dot(a,b)
# matirx_product_alt = a @ b
# print(matrix_product)
# # Determinant
# print(np.linalg.det(a))
# # inverese
# print(np.linalg.inv(a))
# # eigenvalues and eigenvectors
# eigenvalues,eigenvectors = np.linalg.eig(a)
# print(eigenvalues,eigenvectors)

# iterator in numpy
# arr = np.array([[1,2,3],[3,4,8],[9,4,0]])
# print(arr)
# for i in arr:
#     for j in i:   # inefficient way.
#         print(j)
# for i in np.nditer(arr):  # most efficient way.
#     print(i)

#Probability distrubution in random numbers
# arr = np.array([3,5,8,1,4])
# x = np.random.choice(arr,p=[0.1,0.2,0.5,0.2,0.0]) # sum of probability should be 1
# print(x)

# ufunc or universal functions
# They are special function that perform vectorization.
# Vectorization is a process of converting iterative statments into vector based operation is called vectorization.
# example 
# arr1 = np.array([1,2,3])
# arr2 = np.array([4,4,6])
# for i,j in zip(arr1,arr2):
#     print(i+j)            # This is an iterative approach
# arr3 = np.add(arr1,arr2)    # This is ufunc type of function
# print(arr3)
# All arthimatict ufunc
# add()
# subtract()
# multiply()
# divide()
# power()
# mod() or reminder() # both return  remainder 
# divmod()            # return quotient
# abs()

