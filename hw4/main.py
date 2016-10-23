
"""
def smoothCubic_Dec():
	pass

n : the number of points in the mesh, n must be at least 2
p : strictly increasing array of len n 
v: an array of values of the function at the poitns of p
dl: a value that gives g'(p[0])
dr: a value tht gives g'(p[n-1])

"""


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



