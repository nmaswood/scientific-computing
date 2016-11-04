import numpy as np
import matplotlib.pyplot as plt

"""
I have been away at onsites most of this week and previous week
so I have really no clue what is going on. That being said I tried my best!

I used the following resources to try and piece together an understanding of
what this assignment is about:

http://www.cems.uvm.edu/~tlakoba/math337/notes_3.pdf

http://www.dam.brown.edu/people/sahn/html/ab4.html
 ** This resource had the code that helped me figured out how
 to set the initial three values


I get the idea behind Euler's method and single step estimation:
estimate the curve stepwise using a slope at a time however I am
unsure how Adam's method and multistep actually works.

What I tried to do was get the first three values using RK2 and 
use the formula I found in the first link above to try and estimate
the rest of the values

Ya Really no clue what is going on, will probably just try to extra credit
to make this one up at some point... :(

"""

class Solve():

    def __init__(self, f, t0, y0, h):

        N = round (1 / h)
        tn = np.array([h * i for i in range(N)])
        yn = np.zeros(N)
        yn[0] = y0
        t = t0
        y = y0

        for i in range(1,3):
            k1 = f(t,y)
            k2 = f(t + h /2,y + h/2 * k1)
            k3 = f(t + h /2,y + h/2 * k2)
            k4 = f(t+ h,y +h * k3)
            y = y+h/6*(k1 + 2*k2 + 2*k3 + k4)

            t += h
            yn[i] = y

        self.N = N
        self.tn = tn
        self.yn = yn
        self.h = h

        self.f_n = yn[0]
        self.f_n_1 = yn[1]
        self.f_n_2 = yn[2]

        self.n = 3

    def advance(self):

        h = self.h
        n = self.n
        self.yn[n + 1] = f(self.tn[n], self.yn[n])

        new_f = self.yn[n] + h/12. * (23*self.f_n - 16 * self.f_n_1 + 5*self.f_n_2)

        self.f_n_1, self.f_n_2, self.f_n = self.f_n, self.f_n_1, new_f

        self.n +=1

    def main(self):

        for _ in range(self.N - 4):
            self.advance()

        plt.plot(self.tn, self.yn)
        plt.xlabel = 'Energy of orbit?'
        plt.show()
        return (self.tn, self.yn)

"""

  r'' = -r / pow( |r|, 3)

So that we can compare answers, try this with initial
position r(0) = (1,0) and r'(0) = (0, 0.3). Track the total
energy of the system (which should be constant):

   E = 0.5 sq(|r'|) - 1/|r|.

"""


f = lambda x,y: -x /(abs(x) ** 3)

t0 = 1
y0 = 1
h = 0.05

x = Solve(f,t0,y0,h)
res = x.main()