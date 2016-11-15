import numpy as np
from matplotlib.pyplot import plot, show, legend,title, savefig
from scipy.optimize import newton

class System():

    """

    Implementation and Explanation:

    System is a struct that holds all the information necessary to make your
    estimation of the ODE system. It contains the three fucntions f_x,f_y, f_z
    as well as their initial starting points. Furthermore it has parameters for
    dt and steps.

    This class implements a solution of the Belosov Zhabotinksy model using the reverse
    Euler method. The reverse Euler method appears from lines 50 to 60. In order to save time/space
    I used scipy's implementation of Newton's method instead of writing my own.

    Because the System is implemented as general with respect to an ODE system, BZ is the class
    that specfically solves the problem at hand. At first I could not get a reasonable result using
    the values specified by Dupont in the write up, so I tried playing around with other values and found
    a verison of BZ on the internet that used the following equations and values and so I used those values
    and got what I think is a better result? Honestly I don't know. At anyrate here it is 

    https://github.com/KOlexandr/Modeling.Lab5/

    If you want to see what it is like with the original values provided by Dupont
    comment out line 141 and uncomment out line 140, then hit graph.

    """

    def __init__(self, f_x, f_y, f_z, x_0, y_0, z_0, dt = 0.001, steps = 1000):

        self.f_x = f_x
        self.f_y = f_y
        self.f_z = f_z
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.dt = dt
        self.steps = steps

    def unpack(self):

        return (
            self.f_x,
            self.f_y,
            self.f_z,
            self.x_0,
            self.y_0,
            self.z_0,
            self.dt,
            self.steps)

    def solve(self):

        f_x, f_y, f_z, x_0, y_0 , z_0, dt, steps = self.unpack()
        t,x,y,z = [np.zeros(steps) for _ in range(4)]
        x[0],y[0],z[0] = x_0, y_0,z_0

        for i in range(0, steps-1):

            f_1 =  lambda x: x - dt * f_x(x + dt, y[i], z[i])
            f_2 =  lambda y: y - dt * f_x(x[i], y+ dt, z[i])
            f_3 =  lambda z: z - dt * f_x(x[i], y[i], z + dt
                )

            x[i+1] = x[i] + newton(f_1, x[i])
            y[i+1] = y[i] + newton(f_2, y[i])
            z[i+1] = z[i] + newton(f_3, z[i])
            t[i+1] = t[i] + dt

        return t, x, y, z

    def v(self,t,x,y,z):

        f = lambda i: (abs(x[i])/ 125e5) + abs(y[i])/1800 + abs(z[i]) / 3e4

        t_max = max(t)

        max_v = float("-inf")

        for i in range(len(x)):

            max_v = max(max_v, f(i))

        mav_v_700 = float("-inf")

        for i in range(500, 700):

            max_v_700 = max(mav_v_700, f(i))

        return [("%.3f" % q) for q in (max_v, t_max, max_v_700)]

    def plot(self):

        t,x,y,z = self.solve()

        max_v,t_max, max_v_700 = self.v(t,x,y,z)

        for (index,(var, color, label)) in enumerate(((x, 'r', 'x'), (y, 'g', 'y'), (z, 'b', 'z'))):
            plot(t, var, color, label = "f_{}: {} of time".format(index, label))


        legend(loc='best', numpoints=1)
        title("steps:{} dt: {}\n v_max: {} t_max: {} v_max_500: {}".format(self.steps, self.dt, max_v, t_max,max_v_700, ))

        savefig("BZ-{}.png".format(self.dt))
        show()


class BZ():

    def __init__(self, x_0, y_0,z_0,l,c,dt,steps):
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.l = l
        self.c = c
        self.dt = dt
        self.steps = steps

    def unpack(self):

        return self.x_0, self.y_0, self.z_0, self.l, self.c

    def main(self):

        x_0, y_0, z_0, l, c = self.unpack()

        f_0 = lambda y_0, y_1, y2: 77.27 * (y_1 - (y_0 * y_1) + y_0 - (8.375e-6 * (y_0**2)))
        f_1 = lambda y_0, y_1, y2: (1/77.27) * (-y_1 - (y_0 * y_1) + y_2)
        f_2 = lambda y_0, y_1, y2: 0.161 * (y_0 - y_2)

        f_3 = lambda x, y, z: l[0] * y * (c - x) - l[2] * x
        f_4 = lambda x, y, z: l[0] * y * (c - x) - l[1] * y * z + l[4]
        f_5 = lambda x, y, z: l[2] * x + l[5] * (l[6] * y - l[7]) * (l[6] * y - l[7]) * x - l[3] * z

        bz_system = System(
            #f_0, f_1, f_2,
            f_3,f_4,f_5,
            x_0, y_0, z_0, self.dt, self.steps
        )

        bz_system.plot()


x_0, y_0, z_0, l, c, = (4, 1.1, 4, [0.1, 0.5, 1, 1.5, 0.91, 1, 1, 1.9], 15)

for dt in (.1,0.01, 0.001, 0.0001, 0.00001):

    BZ(x_0,y_0,z_0,l,c,dt = dt,steps = 700).main()


