import numpy as np

from numpy import matmul
from scipy.linalg import solve_triangular, solve

# https://github.com/BabisK/M36105P

# Ax = B

class Solver():

    def __init__(self, A,B):
        self.A = A
        self.B = B

    def houseHolderTransf(self,x):

        l = len(x)
        I = np.identity(l)
        b = np.zeros(l)

        b[0] = -np.sign(x[0])*np.linalg.norm(x)

        a_err = x - b
        a_err = a_err.reshape(l,1)

        return I + np.dot(a_err,np.transpose(a_err))/(b[0]*a_err[0])

    def QR(self, A):

        """This function calculate the factorization A = QR"""

        m,n = A.shape
        Q = np.identity(m)
        R = A.copy()

        for i in range(n):
            P = self.houseHolderTransf(R[-m+i:,i])
            Qi  = np.identity(m)
            Qi[-len(P):,-len(P):] = P
            R = np.dot(Qi,R)
            Q = np.dot(Q,Qi)
        return Q,R

    def householder(self):

        Q, R = self.QR(self.A)
        c = matmul(Q.T, self.B)

        return solve_triangular(R, c)       

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

        result = Solver(x,y).householder()
        
        print (from_scipy)
        print (result)

run = Test()
run.main()