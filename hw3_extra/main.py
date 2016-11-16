import numpy as np
from scipy.linalg import solve_triangular, solve

"""
TLDR: The sanity check that this works is to run this script. Two vectors will be printed and they should be the same
one is from  scipy's built in linear equation solver and the other is my implemenation they are working with the same
Ax = B so they should be the same.

The purpose of this project is to produce
a solver that is based on a QR factorization of a
square full matrix.

My solution works in three stages:

1. First I wrote a function that transforms a matrix using the householder method
2. Then I used that function to implement QR factorization
3. Finally, I used the QR form of matrix A in order to solve the matrix
using the triangular solver we developed in hw3. However I do not actually use the solver because I figure just
using scipy is easier and more straightforward.


This site helped me figureo out the methdology.
http://www.seas.ucla.edu/~vandenbe/133A/lectures/qr.pdf

This repo helped me with the code
https://github.com/BabisK/M36105P
"""
class Householder():

    def __init__(self, A,B):

        self.A = A
        self.B = B

    def transformation(self,x):
        l = len(x)

        b = np.zeros(l)

        b[0] = np.linalg.norm(x) * -np.sign(x[0])

        err = (x - b).reshape(l,1)

        transpose_term = b[0] * err[0]

        Y = np.dot(err,np.transpose(err))

        return np.identity(l) + Y/transpose_term

    def QR(self, A):

        m,n = A.shape
        Q = np.identity(m)
        R = A.copy()

        for i in range(n):
            P = self.transformation(R[-1 *m+i:,i])
            l_p = -len(P)
            Q_i  = np.identity(m)
            Q_i[l_p:,l_p:] = P
            R = np.dot(Q_i,R)
            Q = np.dot(Q,Q_i)
        return Q,R

    def solve(self):

        Q, R = self.QR(self.A)
        
        return solve_triangular(R, np.matmul(Q.T, self.B))

class Test():

    def __init__(self):
        self.x = np.array([
            [1,2,1, -1],
            [3 ,2 ,4 ,4],
            [4 ,4 ,3 ,4],
            [2 ,0 ,1 ,5]
            ])

        self.y = np.array([5, 16, 22, 15])

    def main(self):

        x,y = self.x, self.y

        from_scipy = solve(x,y)

        result = Householder(x,y).solve()

        print (from_scipy)
        print (result)

run = Test()
run.main()
