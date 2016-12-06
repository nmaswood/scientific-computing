import numpy as np
from ode import stiff_solver

def construct_df(t,y):

    """

    This is the df matrix as copied from Dupont's example code.

    """

    df = np.zeros((3,3))

    df[0][0] = 77.27*(1.0 - y(1) -2.*8.375e-6*y(0))
    df[0][1] = 77.27*(1.0 -y(0) )
    df[0][2] = 0.0;
    df[1][0] = -1.0/77.27;
    df[1][1] = (-1.0/77.27)*(1.0+y(0))
    df[1][2] = 1.0/77.27
    df[2][0] = 0.161
    df[2][1] = 0.0
    df[2][2] = -0.161

    return df

def ret_f(t,y):

    """

    "fa measures the size of the stuff used in computing f"

    """

    f = np.zeros(3)
    f[0] = 77.27*(y(1) - y(0)*y(1)+ y(0)-8.375e-6*y(0)*y(0))
    f[1] = (1.0/77.27)*(-y(1)-y(0)*y(1)+y(2))
    f[2] = 0.161*(y(0)-y(2))

    return f

def jacobin(y):

    """

    This is the jacobin matrix as copied from Dupont's example code.

    """

    df = np.zeros((3,3))

    df[0,0] = 77.27*(1.0 - y(1) -2.*8.375e-6*y(0))
    df[0,1] = 77.27*(1.0 -y(0) )
    df[0,2] = 0.0;
    df[1,0] = -1.0/77.27;
    df[1,1] = (-1.0/77.27)*(1.0+y(0))
    df[1,2] = 1.0/77.27
    df[2,0] = 0.161
    df[2,1] = 0.0
    df[2,2] = -0.161

    return df

def f(t, y):
    """
    Random testing function
    """
    return (5 * y, 5 * y)   

if __name__ == "__main__":

    # Random testing parameters
    y0_this, t0 = np.array([1.0, 2.0]), np.array([0,.1,.2,.3,.4])

    # Use of solver
    res = stiff_solver(f,y0_this, t0)