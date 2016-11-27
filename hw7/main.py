import numpy as np
import math
from scipy.linalg import solve, norm
from scipy.sparse.linalg import cg

from math import *
from scipy.sparse import csgraph

# http://www.math.iit.edu/~fass/477577_Chapter_16.pdf
# http://georgioudakis.com/blog/categories/python/cg.html

class ConjugateGradientBasic():

  def __init__(self, tolerance, error):

    self.tolerance = tolerance
    self.error = error

  def solve(self, A,B, X):

    rows, _ = A.shape

    r = B - np.dot(A, X)
    pi1 = r

    for i in range(rows):

      r_DOT_r = np.dot(r,r)
      A_DOT_pi1 = np.dot(A,pi1)

      alpha =  r_DOT_r / np.dot(pi1,A_DOT_pi1)
      X += np.dot(alpha, pi1)
      r1 = r - np.dot(alpha,A_DOT_pi1)
      beta =  np.dot(r1,r1) / r_DOT_r
      pi1 = r1 + beta * pi1
      r = r1

    return X

x  = np.array([
  [4,1],
  [1,3]
])

y  = np.array(
  [1,2]
)

res = cg(x,y)
#print (res[0])

r = ConjugateGradientBasic(tolerance = 1e-6, error = 10)

res = r.solve(x,y, np.zeros(x.shape[0]))
######################################################################################################

class ConjugateGradientLapalcian():

  def __init__(self, tolerance, error):

    self.tolerance = tolerance
    self.error = error

  def solve(self, A,B, X, steps):

    rows, _ = A.shape

    r = B - np.dot(A, X)
    pi1 = r

    for _ in range(steps):

      r_DOT_r = np.dot(r,r)
      A_DOT_pi1 = np.dot(A,pi1)

      alpha =  r_DOT_r / np.dot(pi1,A_DOT_pi1)
      X += np.dot(alpha, pi1)
      r1 = r - np.dot(alpha,A_DOT_pi1)
      beta =  np.dot(r1,r1) / r_DOT_r
      pi1 = r1 + beta * pi1
      r = r1

    return X

def sinsin(x,y):
  return sin(pi*2*x)*sin(pi*3*y);

def bumpbump(x,y):
  return -2*(x*(1-x)+y*(1-y));

def zero(x,y):
  return 0

def fill(u, f, dx):
  rows, cols = u.shape

  for i  in range(rows):
    u[i][0] = u[i][rows-1] = u[0][i] = u[N-1][i] = 0.0

  for  i in range(rows):
    for j in range(cols):
      u[i][j] = f(i * dx, j * dx)

def Cu(u):
  rows, cols = u.shape

  retMatrix = np.zeros((rows,cols))

  for i in range(1,rows-1):
    for j in range(1,cols-1):
      retMatrix[i][j] = sqrt(N-1) * (4 * u[i][j] \
        - u[i-1][j] - u[i+1][j] - u[i][j-1] - u[i][j+1])

  return retMatrix

if __name__ == "__main__":
  print ("bar")
  N = 10
  u = np.zeros((N,N))
  r = np.zeros((N,N))
  dx = 1.0 / (N - 1)
  fill(r, sinsin,dx)

  normSquared = norm(r)
  print (normSquared)

  CuMatrix = Cu(r)
  print (CuMatrix)

  # Matrix s(r)

  output1 = "Test case 1: The initial norm squared {}".format(normSquared)

  fill(r, bumpbump, dx)

  s = r

  normRessq = norm(r)

  output2 = "Test case 2: The initial norm squared of the residual is {}".format(normRessq)
















#print (csgraph.laplacian(x, normed=False))



