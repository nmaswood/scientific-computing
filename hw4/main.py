from math import sin, cos
import numpy as np
"""

Explanation:

I implemented natural, clamped and not-a-knot. 

I am not sure that I got them right, however I think I have the overall idea somewhere.

There are two functions at the bottom part_one and part_two and when you run them 

the derivates for each cubic portion and the error (for which I did actual - estimated for some xs)

will be returned.

sources: 

http://www.math.wsu.edu/faculty/genz/448/lessons/l304.pdf
http://facstaff.cbu.edu/wschrein/media/M329%20Notes/M329L67.pdf
http://www.physics.arizona.edu/~restrepo/475A/Notes/sourcea-/node35.html

"""

######################### FROM HOMEWORK 3 ########################################

"""

I used the FullSystem class to solve a Ax = b for the not-a-knot case.

"""

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

#################################################################################

class Function():


	"""
	This class is a wrapper around a function it has its 
	implementation ,range, values for its derivates at n_0 and n_last
	and the function correspoding its derivative
	"""

	def __init__(self, f, start, end, d1, d2, n, derivative = None):
		"""
		f : Function
		Function we are interpolating

		start: Int
		The start of our range

		end : Int
		Then end of our range

		d1: Float
		The  Value of the first derivative 

		d2: Float

		The value of the second derivative

		n : Int
		numbers in range

		derivative : Function

		dx/dy of function

		"""

		self.f = f
		self.start = start
		self.end = end 
		self.d1 = d1
		self.d2 = d2
		self.n = n
		self.derivative = derivative

	def calculate_range(self):
		"""

		Creates a domain of values from (start -> end)
		Evaluate f(x)  at each of these values

		returns xs and ys

		"""

		xs = np.linspace(self.start,self.end, self.n)

		ys = [self.f(x) for x in xs]

		return (xs,ys)	

class Natural():

	"""

	This class implements cubic interpolation in the natural case
	where the derivate at x_0 and the derivate at x_n is unconditonally
	set to zero. I don't think we actually had to implement this, but
	it seemed like a normal progression between natural, clamped and 
	not-a-knot so I did anyway

	This algorithim is from http://facstaff.cbu.edu/wschrein/media/M329%20Notes/M329L67.pdf

	"""

	def __init__(self,n, xs, ys):


		"""
		n: Int
		s the number of points you are trying to interpolate

		xs: List Float
		The x coordinates

		ys: List Float
		The y coorindates

		"""

		self.n  = n
		self.xs = p
		self.ys = v

	def unpack(self):
		"""

		This is a convience function that returns the paramters
		of the function as a tuple.

		"""

		return (self.n, self.xs, self.ys)

	def evaluation(self):
		"""
		This function performs the evaluation of a natural cubic spline.
		It builds up the three relevant rows in our matrix and then performs
		triangular row-reduction in order to give us our final list of tuples
		which correspond to the cofficents of our cubic function at every point
		"""

		n, xs, a = self.unpack()

		# Step 1

		h = [xs[i+1] - xs[i] for i in range(n-1)]

		# Step 2

		calculate_alpha = lambda i:  ((3 / h[i]) * (a[i+1] - a[i])) - (3 / h[i-1]) * (a[i])

		alpha = [calculate_alpha(i) for i in range(n -1)]

		# Step 3

		l = np.zeros(n); l[0] = 1

		mu = np.zeros(n); mu[0] = 0

		z = np.zeros(n); z[0] = 0

		# Step 4

		for i in range(1,n - 1):

			l[i] = (2 *  (xs[i+1] - xs[i-1])) - (h[i-1] * mu[i-1])
			mu[i] = h[i] / l[i]
			z[i] = (alpha[i] - (h[i-1] * z[i-1]))  / l[i]

		# Step 5

		l[n-1] = 1
		z[n-1] = 0
		z[n-1] = 0

		# Step 6

		for j in range(n-1, -1, -1):

			c[j] = z[j] - (mu[j] * c[j + 1])
			b[j] = ((a[j+1] - a[j]) / h[j]) - h[j] * (c[j+1] + 2 * c[j]) / 3
			d[j] = (c[j+1] - c[j]) / (3 * h[j])

		# Step 7 

		return list(zip(a,b,c,d))

class Clamped():

	"""

	This class implements cubic interpolation in the clamped case
	where the derivate at x_0 is FPO and the derivate at x_n iss FPN.

	http://www.physics.arizona.edu/~restrepo/475A/Notes/sourcea-/node35.html

	"""

	def __init__(self,n, xs, ys, FPO, FPN):

		"""

		n: Int
		Is the number of points you are trying to interpolate

		xs: List <Floats>
		The x coordinates

		ys: List <Floats>
		The y coorindates corresponding to the x coorindates

		FPO: Float
		the derivative at point x_0

		FPN: Float
		the derivate at point x_n

		"""

		self.n  = n
		self.xs = xs
		self.a = ys
		self.FPO = FPO
		self.FPN = FPN

	def unpack(self):

		"""

		returns paramters of function as a tuple

		"""

		return (self.n, self.xs, self.a, self.FPO, self.FPN)

	def evaluation(self):

		n, xs, a, FPO, FPN = self.unpack()

		# Step 1

		h = np.zeros(n)

		for i in range(n-1):

			h[i] = xs[i+1 ] - xs[i]

		# Step 2

		alpha = np.zeros(n)

		alpha[0] = (3 * (a[1] - a[0])) - (3 * FPO)

		alpha[n - 1] = (3 * FPN) - (3 * (a[n-1] - a[n-2]) / h[n-2])


		# Step 3


		for i in range(1, n-1):

			alpha[i] = (
				(3 / h[i]) * (a[i+1] - a[i])  - 
				(3 / h[i-1]) * (a[i] - a[i-1])
			)

		# Step 4

		l = np.zeros(n)
		l[0] = 2 * h[0]

		mu = np.zeros(n); mu[0] = 0.5
		z = np.zeros(n); z[0] = alpha[0] / l[0]


		# Step 5

		for i in range(1,n-1):

			l[i] = (2 * (xs[i+1] - xs[i-1])) - (h[i-1] * mu[i-1])

			mu[i] = h[i] / l[i]

			z[i] = (alpha[1] - (h[i - 1]*z[i-1])) / l[i]

		# step 6

		l[n-1] = (h[n-2]) * (2 - mu[n-1])
		z[n-1] = (alpha[n-1] - (h[n-2] * z[n-1])) / l[n-1]

		c = np.zeros(n)

		c[n-1] = z[n-1]

		# step 7

		b = np.zeros(n)
		d = np.zeros(n)

		for j in range(n-2, -1,-1):
			c[j] = z[j] - (mu[j] * c[j+1])
			b[j] = ((a[j+1] - a[j]) / h[j]) - ((h[j] * (c[j+1] * 2 * c[j]) / 3))
			d[j] = (c[j + 1] - c[j]) / (3 * h[j])

		return list(zip(a,b,c,d, xs))

	def derivative(self, a, b, c, d, x_i, x):
		"""

		Calculataes the deriative of a cubic function given 
		its coefficents and x_i and x

		"""

		diff = x - x_i

		return sum(
			(
				2 * a * (diff) ** 2,
				2 * b * diff,
				c
				)
			)

	def return_derivatives(self, FunctionObject):
		"""
		returns the deriviatives of function over a range
		"""

		xs, ys = FunctionObject.calculate_range()

		n, d1, d2 = x_cubed.n, x_cubed.d1, x_cubed.d2

		res = self.evaluation()

		results = []

		actual = []

		for (x, (a,b,c,d,x_i)) in zip(xs, res):

			l = self.derivative(a,b,c,d,0, x)

			k = FunctionObject.derivative(x )

			results.append(l)
			actual.append(k)

		return (results, actual)
	

x_cubed = Function(
	lambda  x: x  ** 3,
	start = 0,
	end = 10,
	d1 = 0,
	d2 = 300,
	n = 10,
	derivative = lambda x :  3 * (x ** 2)
	)

sin_f = Function(
	sin,
	start = 0,
	end = 10,
	d1 = 0,
	d2 = 300,
	n = 10,
	derivative = cos
	)


class NotAKnot():

	def __init__(self, n,p,v):

		self.n = n
		self.xs = p
		self.ys = v

	def unpack(self):

		return (self.n, self.xs, self.ys)

	def evaluation(self):

		n, xs, ys = self.unpack()

		h = []; g = []; g_prime = []; H = []

		for i in range(n-1):
			h.append(xs[i+1] - xs[i])

		for i in range(n-1):

			g.append(ys[i + 1] - ys[i])


		for i in range(1,n-1):

			g_prime.append(6 * (g[i] - g[i-1]))

		# build matrix 
		for i in range(len(h) - 1):

			r1 = [0 for t in range(i)] 
			r2 = [h[i],2*(h[i]+h[i+1]),h[i+1]]
			r3 = [0 for t in range(len(h)-2-i)]

			total_row = r1 + r2 + r3

			H.append(total_row)

		G = [0] + g_prime + [0]

		H = [[h[1],-h[1]-h[0],h[0]] + [0 for t in range(len(H[0])-3)]] + H + [[0 for t in range(len(H[0])-3)] + [h[-1],-h[-1]-h[-2],h[-2]]]

		# Solve the system of linear equations 

		result = FullSystem(H,G).solve()

		# corresponds to a,b,c,d and xs respectively

		a_s = []; b_s = []; c_s = []; d_s = [];

		for i in range(n-1):

			a_s.append(result[i+1]/(6*h[i]))
			b_s.append(result[i]/(6*h[i]))
			c_s.append((ys[i+1]/h[i])-(result[i+1]*h[i]/6))
			d_s.append((ys[i]/h[i])-(result[i]*h[i]/6))


		return list(zip(
			a_s,
			b_s,
			c_s,
			d_s,
			xs
		))

	def derivative(self, a, b, c, d, x_i, x):
		"""

		Calculataes the deriative of a cubic function given 
		its coefficents and x_i and x

		"""

		diff = x - x_i

		return sum(
			(
				2 * a * (diff) ** 2,
				2 * b * diff,
				c
				)
			)

	def return_derivatives(self, FunctionObject):
		"""
		returns the deriviatives of function over a range
		"""

		xs, ys = FunctionObject.calculate_range()

		n, d1, d2 = x_cubed.n, x_cubed.d1, x_cubed.d2

		res = self.evaluation()

		results = []

		actual = []

		for (x, (a,b,c,d,x_i)) in zip(xs, res):

			l = self.derivative(a,b,c,d,0, x)

			k = FunctionObject.derivative(x )

			results.append(l)
			actual.append(k)

		return (results, actual)


def part_one():

	xs, ys = x_cubed.calculate_range()
	n, d1, d2 = x_cubed.n, x_cubed.d1, x_cubed.d2
	
	c = NotAKnot(n,xs, ys)

	cubed_res = c.return_derivatives(x_cubed)

	xs, ys = x_cubed.calculate_range()

	n, d1, d2 = sin_f.n, sin_f.d1, sin_f.d2

	res =  Clamped(n, xs, ys, d1, d2)

	estimation, actual = res.return_derivatives(sin_f)
	error = sum([abs(estimation_i - actual_i) for (estimation_i, actual_i) in zip(estimation, actual)] )

	return (cubed_res, error)


def part_two():

	xs, ys = x_cubed.calculate_range()
	n, d1, d2 = x_cubed.n, x_cubed.d1, x_cubed.d2
	
	c = NotAKnot(n,xs, ys)

	cubed_res = c.return_derivatives(x_cubed)

	xs, ys = x_cubed.calculate_range()

	n, d1, d2 = sin_f.n, sin_f.d1, sin_f.d2

	res =  Clamped(n, xs, ys, d1, d2)

	estimation, actual = res.return_derivatives(sin_f)
	error = sum([abs(estimation_i - actual_i) for (estimation_i, actual_i) in zip(estimation, actual)] )

	return (cubed_res, error)

part_one()
part_two()
