from scipy.linalg import solve 
import numpy as np
import matplotlib.pyplot as plt

class Input():

	"""

	This the data structure holds all the information for each different variable.
	Each represents an example from one of those ten things on the homework.


	"""

	def __init__(self,a,b, sigma, k, v, g, f, y_0, y_1, sol, name):
		self.a = a
		self.b = b
		self.sigma = sigma
		self.k = k 
		self.v = v
		self.g = g
		self.f = f 
		self.y_0 = y_0
		self.y_1 = y_1
		self.sol = sol
		self.name = name

	def unpack(self):

		i = self

		return (
			i.a,
			i.b,
			i.sigma,
			i.k,
			i.v,
			i.g,
			i.f,
			i.y_0,
			i.y_1,
			i.sol,
		)

class Solution():

	def __init__(self, Input):
		self.Input = Input

	def solve(self):

		"""

		This function implements the Galerkin method to the best of my
		understanding. The picture of Todd Dupont's whiteboard in the folder
		is where I got the idea for this implementation. In addition I provided
		as much comments as I could in code.

		"""

		# This line just unpacks all the input variables

		a,b,sigma,k,v,g,f, y_0, y_1, sol = self.Input.unpack()

		# This the size your final matrix
		# your matrix is l_sigma x l_sigma

		l_sigma = len(sigma)

		# we are esssentially solving Ax = b
		# This matrix is A
		matrix = np.zeros((l_sigma, l_sigma))

		# This matrix keeps track of the delta x
		# between different iterations

		dx = [sigma[i+1] - sigma[i] for i in range(l_sigma - 1)]

		prev = dx[0]

		# This matrix is b in Ax=b
		b = []


		for (index, x_i) in enumerate(sigma[:-1]):

			dx_value = 1/(x_i - prev)

			# This is the midpoint
			m = (x_i + sigma[index + 1]) / 2

			# These are teh four equations as outlined on the board by Dupont

			i_00 = (k(x_i - .5 )) * (dx_value) + v (x_i - .5) * -.5 + g(x_i - .5) * (dx_value / 3)
			i_01 = (k(m)) * (-dx_value) + v (m) * -.5 + g(m) * (dx_value / 6)
			i_10 = (k(m)) * (-dx_value) + v (m) * -.5 + g(m) * (dx_value / 6)
			i_11 = (k(m)) * (dx_value) + v (m) * -.5 + g(m) * (dx_value / 3)

			# I really don't know how to properly due with 
			# r_0 and r_1 but I assuming that they are x_i
			# in the x of Ax = b

			r_0 = f(x_i - .5) / (dx_value /2)

			r_1 = f(m)  * 1 / (dx_value) / 2

			prev = x_i

			b.append(r_1)

			matrix[index][index] = i_00
			matrix[index+1][index] = i_01
			matrix[index][index+1] = i_10
			matrix[index+1][index+1] = i_11

		# this is to satisfy one of the boundary conditions

		b.append(y_1)

		res = solve(matrix, b)	

		return sigma, res
	
# Example 1 Take a = 0, b = 1, σ = {0, 0.2, 0.7, 0.8, 1}, k = 1, v = g = f = 0,
# y(0) = 0, y(1) = 1. The solution to the differential problem and the Galerkin
# problem is y(x) = x.

Example1 = Input(
	a = 0,
	b = 1,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 1,
	v = lambda x: 0,
	g = lambda x: 0,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 1'
)

Example2 = Input(
	a = 0,
	b = 1,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 1,
	v = lambda x: 0,
	g = lambda x: 0,
	f = lambda x: 2,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 2'
)

Example3 = Input(
	a = 0,
	b = 1,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 4 if x <= .7 else 1,
	v = lambda x: 0,
	g = lambda x: 0,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 3'
)

Example4 = Input(
	a = 0,
	b = 1,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 1,
	v = lambda x: 5,
	g = lambda x: 0,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 4'
)

Example5 = Input(
	a = 0,
	b = 1,
	sigma = [0,0.25,.5,.75,1],
	k = lambda x: 1,
	v = lambda x: 5,
	g = lambda x: 0,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 5'
)	

Example6 = Input(
	a = 0,
	b = 1,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 1,
	v = lambda x: 0,
	g = lambda x: 5,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 6'
)

Example7 = Input(
	a = 0,
	b = 1,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 1,
	v = lambda x: 0,
	g = lambda x: 5,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 7'
	)

Example8 = Input(
	a = 1,
	b = 0,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 1,
	v = lambda x: 0,
	g = lambda x: 5,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol  = lambda x: x,
	name = 'Example 8'
	)

Example9 = Input(
	a = 1,
	b = 0,
	sigma = [0,0.2,.7,.8,1],
	k = lambda x: 1,
	v = lambda x: 0,
	g = lambda x: 5,
	f = lambda x: 0,
	y_0 = 0,
	y_1 = 1,
	sol = lambda x: x,
	name = 'Example 9'
	)

Example10 = Input(
	a = 0,
	b = 2,
	sigma = [i/20  for i in range(20)],
	k = lambda x: 1,
	v = lambda x: 0,
	g = lambda x: 5,
	f = lambda x: x-1,
	y_0 = 0,
	y_1 = 1,
	sol = lambda x: x,
	name = 'Example 10'
	)


class Plot():

	def __init__(self):

		pass

	def plot_and_save_single(self, Input):

		r = Solution(Input)

		xs, ys = r.solve()

		plt.plot(xs,ys)

		plt.savefig('{}.png'.format(Input.name))


	def plot_all(self):

		examples = (Example1,Example2,Example3,Example4,Example5,Example6,Example7,Example8,Example9,Example10)

		for example in examples:

			self.plot_and_save_single(example)

p = Plot()
p.plot_all()