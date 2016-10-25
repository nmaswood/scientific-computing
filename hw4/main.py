
"""
def smoothCubic_Dec():
	pass

n : the number of points in the mesh, n must be at least 2
p : strictly increasing array of len n 
v: an array of values of the function at the poitns of p
dl: a value that gives g'(p[0])
dr: a value tht gives g'(p[n-1])

"""

class Natural():

	"""
	Assuming that S''(x_0) = S''(x_n) == 0
	"""

	def __init__(self,n, xs, ys):


		"""
		n: Is the number of points you are trying to interpolate
		xs: The x coordinates
		ys: The y coorindates

		"""

		self.n  = n
		self.xs = p
		self.ys = v

	def unpack(self):

		return (self.n, self.xs, self.ys)

	def evaluation(self):

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






class HermiteCubic():

	def __init__(self,n, p, v, dl, dr):

		self.n  = n
		self.p = p
		self.v = v
		self.dl = dl
		self.dr = dr

	def evaluation(self):

		pass


class NotAKnot():

	def __init__(self, n,p,v):

		self.n = n
		self.n = p
		self.n = v


class Function():

	def __init__(self, f, d1, d2, of_one):

		self.f = f
		self.d1 = d1
		self.d2 = d2
		self.of_one = of_one
"""
v_o  = Function(
	lambda x: 1 - (3 * x ** 2) + (2 *  x  ** 3),
	-6,
	12,
	6)

v_1 = Function(
	lambda x: (3 * (x  ** 2)) - (2 * (x ** 3)),
	6,
	-12,
	-6)

s_0 = Function(
	lambda x: ((x  - 1) **2) * x ,
	-4,
	-,
	-6)

s_1 = Function(
	lambda x: (x ** 2)  * (x - 1)
	6,
	-12,
	-6)
"""



