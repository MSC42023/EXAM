import numpy as np
import tensorflow as tf

#Create Vector, Matrix and Tensor
x= np.array([1, 2, 3, 4])
print("Create a Vector :", x)
print("\n")

A = np.array([[1, 2], [3, 4], [5, 6]])
print("Create a Mateix:", A)
print("\n")

tensor_A = tf.constant([[1, 2]], dtype = tf.int32)
print("Create a Tensor :", tensor_A)
print("\n")

#Multiplication of tho: Vector, Matrix and
A = np.array([[1, 2], [3, 4], [5, 6]])
print("A=", A)
print("\n")

B = np.array([[2, 5], [7, 4], [4, 3]])
print("B=",B)
print("\n")

C=A*B
print("Multiplication of two Matrix 1", C)
print("\n")

x= np.array([1, 2, 3, 4])

y= np.array([5, 6, 7, 8])

z = x*y 
print("Multiplication of two Vector ", z)
print("\n")

tensor_A = tf.constant([[4, 2]], dtype = tf.int32)
print("A", tensor_A)

tensor_B = tf.constant([[7, 4]], dtype = tf.int32)
print("B:", tensor_B)
tensor_multiply = tf.multiply(tensor_A, tensor_B)
print("Multiplication of two Tensor", tensor_multiply)
print("\n")

#Addition of two: Vector, Matrix and Tensor

x=np.array([1, 2, 3, 4])
y=np.array([5, 6, 7, 8])

z=x+y
print("Addition of two Matrix : ",z)
print("\n")

A=np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[2, 5], [7, 4], [4, 3]])

C=A *B
print("Addition of two vector:", C)
print("\n")

tensor_add = tf.add(tensor_A, tensor_B)
print("Addition of two Tenson", tensor_add)
print("\n")

x = np.array([1, 2, 3, 7, 3, 5, 21])
y = np.array([[1], [3], [5], [7], [8], [8], [2]])
c=x*y
print("Multiply Matrix with Vector :",c)
print("\n")

# Matrix Dot product and Matrix Inverse

U = [2,3]
V = [1, 3]
dotproduct = np.dot (U, V)
print("Matrix Dot product print :",dotproduct)
print("\n")

A=np.array([[6, 1, 1],
[4, -2, 5],
[2, 8, 7]])
# Calculating the inverse of the matrix
print("Inverse of the matrix:", np.linalg.inv(A))
print("\n")
