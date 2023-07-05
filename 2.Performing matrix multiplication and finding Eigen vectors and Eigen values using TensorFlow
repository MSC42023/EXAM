import tensorflow as tf

x = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) 
y= tf.constant([7, 8, 9, 10, 11, 12], shape=[3,2])
print(x)
print(y)

#Multiplying Two matrix
z = tf.matmul(x, y)
print("\n")
print("Multiplying two Matrix :", z)
#  Let's see how we can compute the eigen vectors and values from a matrix

e_matrix_A = tf.random.uniform(
  [2, 2], minval=3, maxval=10, dtype=tf.float32, name="matrixA") 
print("\n")
print("Matrix A:\n{}\n\n".format(e_matrix_A))
#  Calculating the eigen values and vectors using tf-linale.eigh, if you only want the values you can use eigvalsh

eigen_values_A, eigen_vectors_A= tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors: \n{}\n\nEigen Values: \n{}\n".format(
  eigen_vectors_A, eigen_values_A))
