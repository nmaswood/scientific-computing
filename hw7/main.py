import numpy as np
import math
from scipy.linalg import solve
from scipy.sparse.linalg import cg

# http://georgioudakis.com/blog/categories/python/cg.html

class ConjugateGradient():

  def __init__(self, tolerance, error):

    self.tolerance = tolerance
    self.error = error

  def erf(self, matrix_one, matrix_two, n):

    return sum([(matrix_one[i] - matrix_two[i]) ** 2 for i in range(n)]) ** .5

  def solve(self, A,B):

    n, _ = A.shape

    X = np.zeros(n)
    r = B - np.dot(A, X)
    pi1 = r

    for i in range(n):

      if self.error < self.tolerance:
        break

      Y = np.copy(X)
      alph =  np.dot(r,r) / np.dot(pi1, np.dot(A,pi1))
      X = Y + np.dot(alph, pi1)
      r1 = (r - np.dot(alph,np.dot(A, pi1)))
      beta =  np.dot(r1,r1) / np.dot(r,r) 
      pi1 = r1 + beta * pi1
      r = r1
      err = self.erf(Y, X, n)

    return X

x  = np.array([
  [4,1],
  [1,3]
])

y  = np.array(
  (1,2)
)

res = cg(x,y)
print (res[0])

r = ConjugateGradient(tolerance = 1e-6, error = 10)

res = r.solve(x,y)
print (res)