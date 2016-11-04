import numpy as np 

class FullSystem():

    """
    Citations:

    http://www.math.usm.edu/lambers/mat610/sum10/lecture4.pdf


    Ax = B
    In in this struct the first parameter is a and the second parameter is b.

    Calling solve method on one of these classes will return your matrix x

    """

    def __init__(self, a, b):

        self.a = a
        self.b = b 

    def clone(self):
        """

        Simply makes a copy of a,b so that original values
        are not changed when the algorithim modifies them

        """

        return np.copy(self.a), np.copy(self.b)

    def solve(self):
        """
        Solves for the vector x
        """

        return FullGauss().solve(self)

class FullGauss:
    """
    This function implements Gaussian Elmination on a full n x n matrix.
    """

    def reduce(self, system):
        """

        This function puts the nxn matrix in upper trinagular form

        """


        A, B = system.clone()

        n = len(A)

        for j in range (n):
            for i in range(j + 1,n):

                m = A[i][j]  / A[j][j]
                for k in range(j + 1, n):

                    A[i][k] -= (m * A[j][k])

                B[i] -= m * B[j]

        return  FullSystem(A,B)

    def substitute(self, system):

        """
        This function uses back substituion to find the solution vector x
        """

        A,B = system.clone()

        n = len(A)

        xs = np.zeros(n)

        for i in range(n - 1 , -1, -1):

            xs[i] = B[i]

            for j in range(i + 1, n):

                xs[i] -= A[i][j] * xs[j]

            xs[i] /= A[i][i]

        return xs

    def solve(self, system):
        """
        This function puts the matrix into reduced form 
        and then uses back substituion to get the final value

        """

        reduced = self.reduce(system) 

        return self.substitute(reduced)

class TDMSystem():

    """
    A TDM can only have up to three rows of values.
    Therefore there is no point in storing an entire
    array of zeroes if we are only just going to use up to
    three rows.

    Therefore the main diagnol b, the sub diagnol a, or the super diagnol c
    are the only rows stored

    d is the vector b in Ax = b

    """

    def __init__(self, a,b, c, d):

        self.a  = a
        self.b  = b
        self.c  = c
        self.d  = d

    def unpack(self):

        return  (
            np.copy(self.a),
            np.copy(self.b),
            np.copy(self.c),
            np.copy(self.d)
        )

    def solve(self):

        f = TDMGauss()

        reduced = f.reduce(self)
        return f.substitute(reduced)


class TDMGauss():

    """ 
    Citations:


    This is an implemention of Thomas's alogorithim which I read about 
    here: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    In addition to that this file helped me implement the code as well:

    https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
    """

    def reduce(self,  system):

        a,b,c,d =  system.unpack()
        n = len(d)

        for i in range(1,n):

            m = a[i-1] / b[i-1]
            b[i] -= m * c[i-1]
            d[i] -= m * d[i-1]

        return TDMSystem(a,b,c,d)

    def substitute(self, system):

        _,b,c,d =  system.unpack()

        n = len(d)

        result = np.copy(b)
        result[-1] = d[-1]/b[-1]

        for i in range(n - 2, -1, -1):
            result[i] = (d[i] - c[i] * result[i + 1]) / b[i]
        return result

x = np.array([
    [1,2,1, -1],
    [3 ,2 ,4 ,4],
    [4 ,4 ,3 ,4],
    [2 ,0 ,1 ,5]
])

y = np.array([5, 16, 22, 15])

a = np.array([
    [1,0,0,0],
    [0,2,0,0],
    [0,0,3,0],
    [0,0,0,4]
])

b = np.array([5, 16, 22, 15])

A = FullSystem(x,y).solve()
print (A)

a = np.array([3.,1,3]) 
b = np.array([10.,10.,7.,4.])
c = np.array([2.,4.,5.])
d = np.array([3,4,5,6.])


x = TDMSystem(a,b,c,d)
res = x.solve()
print (res)
