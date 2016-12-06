from math import sqrt
from scipy.linalg import lu, solve
from scipy.optimize import newton
import numpy as np

def jacobian(F, x, t):


    ftx = np.array(F(t,x))

    dx = x / 100
    dFdx = (F(t,x + dx) - ftx) / dx

    return dFdx


def stiff_solver(F, y0, tspan, reltol = 1e-5, abstol = 1e-8,initstep=0):

    """

    This is matlab's ode23. I did not implement the inital step solver that their algorithim
    utilized. So the initial step is basically always set at zero. However, I did implement the
    rest of it.

    """

    minstep=abs(tspan[-1] - tspan[1])/1e18
    maxstep=abs(tspan[-1] - tspan[1])/2.5

    d = 1 / (2 + sqrt(2))
    e32 = 6 + sqrt(2)

    t = tspan[0]

    tfinal = tspan[-1]


    h = initstep

    tdir = int(tfinal - t >= 0)


    h = tdir * min(abs(h), maxstep)
    y = y0
    tout = [t]
    yout = [np.copy(y)]


    jac = lambda t, y: fdjacobian(F, y, t)

    J = jac(t,y)

    while abs(t- tfinal) > 0 and minstep < abs(h):

        if abs(t- tfinal) < abs(h):
            h = tfinal - t

        if len(J) == 1:
            W = I - h * d * J
        else:
            W = lu(I - h * d * J)

        T = h * d * (F(t + h /100, y) - F0) (h/100)

        k1 = np.solve(W , F0 + T)

        F1 = F(t + 0.5*h, y + 0.5 * h * k1)
        k2 = np.solve(W , (F1 - k1) + k1)


        ynew = y + h * k2
        F2 = F(t + h, ynew)
        k3 = np.solve(W, (F2 - e32 * (k2 - F1) - 2 *(k1 - F0) + T))


        err = (abs(h)/6)*norm(k1 - 2*k2 + k3)
        delta = max(reltol*max(norm(y),norm(ynew)), abstol)

        if err <= delta:

            for toi in tspan:

                s = (toi - t) / h

                tout.append(toi)
                yout.append(y + h*( k1*s*(1-s)/(1-2*d) + k2*s*(s-2*d)/(1-2*d)))

            if tout[-1] != t+ h:
                tout.append(t + h)
                yout.append(ynew)

            t += h
            y = ynew
            F0 = F2
            J = jac(t,y)

        h = tdir*min( maxstep, abs(h)*0.8*(delta/err)**(1/3))

    return tout, yout