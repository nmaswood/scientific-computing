import matplotlib.pyplot as plt
from math import log, pi, sin, sqrt

### Structs ###


# Function
#
# function -> List Int -> Function
#
# Holds a function to be tested and the interval
# it should be tested on.

class Function():

	def __init__(self, function, interval):

		self.f = function
		self.interval = interval

## Functions to be tested ##

# Function : x^2
# Interval : [0,2]

SQRD = Function(lambda x: x ** 2, [0,2])

# Function : sqrt( sin( pi x/2))
# Interval : [0,1]

SQRT = Function(lambda x: sqrt(sin((pi * x) / 2)), [0,1])

# Function : x^4
# Interval : [-1,1]

FOURTH = Function(lambda x: x ** 4, [-1,1])

# Function : 1 if x<0.5 and 0 if x>=0.5
# Interval : [0,1]


def func4(x):
	if x < 0.5:
		return 1
	elif x >= 0.5:
		return 0

POTENTIAL = Function(func4, [0,1])

# Result
#
# Float -> Int -> Int-> Result
# A tuple of the approximation of integral area
# and the deepest level taken to get there as well
# the number of function calls

class Result():

	def __init__(self, result,function_calls):

		self.result = result
		self.function_calls = function_calls

	def __str__(self):

		return 'Result is {}\nFunction Calls is {}'.format(
			self.result, self.function_calls)


# AdaptiveQuadrature
#
# Implements the adapative quadrature algorithim

class AdaptiveQuadrature():

	def __init__(self):

		# Keeps track of the number function calls
		self.function_calls = 0
		self.i = 10

	def recursive(self, f, tolerance_original, a_original,b_original, max_level, special = False):

		"""

		recursive

		function -> Float -> Float -> Float -> Int -> Bool -> Result

		Calculates the adaptive quadrative for a function
		over an interval. If the special flag is set returns
		instead of the fine approximation and combination of the
		fine and coarse for a better estimate.

		"""

		def sub_routine(tolerance, a, b, level):

			self.i += 1

			f_a, f_b = f(a), f(b)

			coarse = ((f_a + f_b) * (b-a)) / 2

			m = (a+b) / 2

			f_m = f(m)

			first_trap = ((f_a + f_m) * (m - a)) / 2
			second_trap = ((f_m + f_b) * (b - m)) / 2

			fine = first_trap + second_trap

			if abs(fine - coarse) < (3 * tolerance) and level <= max_level:

				self.function_calls += 1

				return fine

			else:

				return  sub_routine(tolerance / 2, a, m, level + 1) + \
				        sub_routine(tolerance / 2, m, b, level + 1)

		approximation = sub_routine(tolerance_original, a_original,b_original, 0)

		return Result(approximation, self.function_calls)

fuck = lambda x: (x + 1) ** 3
run = AdaptiveQuadrature()
res = run.recursive(fuck, 1e-3, 0, 1, 10)
print (res)


# Example
#
# Holds the methods and constants
# used to test and the different
# factors against one another.

class Example():

	def __init__(self):

		self.MAX_LEVEL = 10
		self.PADDING = 1e-30
		self.ACTUAL_ERROR_TOLERANCE = 1e-1
		self.TOLERANCE_RANGE = (5e-1, 3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 5e-3, 3e-3,2e-3,)

	def calculate(self, f, a, b):

		"""

		calculate

		function -> Float -> Float -> Dict<String,Tuple<List Float, List Float>>

		Returns a dictionary containing all the x,y points to be graphed

		"""	


		def function_evaluations_test(f, a, b):

			"""

			function_evalution_test

			function -> Float -> Float -> Tuple <Float,Float>

			Returns a tuple where xs is the number of function calls
			and ys is the log(1/tol)

			"""

			results = [
				AdaptiveQuadrature().recursive(f, tol_variable, a,b, self.MAX_LEVEL) for 
				tol_variable in self.TOLERANCE_RANGE
			]

			xs = [x.function_calls for x in results]

			ys = [log(1/(tol_variable + self.PADDING)) for tol_variable in self.TOLERANCE_RANGE]

			return (xs,ys)

		def actual_error_test(f,a,b):

			"""

			actual error test

			function -> Float -> Float -> Tuple <Float,Float>

			Returns a tuple where xs is ____
			and ys is _____

			"""

			#actual_error = AdaptiveQuadrature().adaptive_quadrature(f,self.ACTUAL_ERROR_TOLERANCE, a,b, self.MAX_LEVEL)

			#xs = [1/ log(actual_error)] * len(self.TOLERANCE_RANGE)

			#ys = []

			return ([1,2,3], [1,2,3])
		def good_interval_length_test(f,a,b):

			"""

			good_interval_length_test

			function -> Float -> Float -> Tuple <Float,Float>

			Returns a tuple where xs is _
			and ys is __

			"""

			return ((1,2,3), (1,2,3))



		return {
			'function_evaluations': function_evaluations_test(f,a,b),
			'actual_error': actual_error_test(f,a,b),
			'good_interval': good_interval_length_test(f,a,b)
		}

	def do_all_tests(self):

		"""
		do_all_tests

		-> Dict< String, Dict<String,Tuple<List Float, List Float>>

		"""

		results = []
		
		for function_struct in (SQRD, SQRT, FOURTH):#,POTENTIAL):

			f = function_struct.f
			a,b = function_struct.interval

			results.append(self.calculate(f,a,b))

		return results


class Plot():


	def __init__(self):
		self.i = 1

	"""

	plot_one

	Float -> Float -> xlabel -> ylabel

	Creates a single plot

	"""

	def plot_one(self,xs, ys, xlabel, ylabel, title):

		foo = plt.subplot(2,2,self.i)
		self.i += 1
		foo.plot(xs, ys)
		foo.set_xlabel(xlabel)
		foo.set_ylabel(ylabel)
		return foo

	def plot_one_function(self, function_name, function_data_dict):

		x = plt.figure(1)
		x.suptitle(function_name)


		for key in ('function_evaluations', 'actual_error', 'good_interval'):

			xs, ys = function_data_dict[key]

			label_dict = {

				'function_evaluations': ('function calls', '1/ log(tol)'),
				'actual_error': ('1 /log(actual_error)', 'log(1/tol)'),
				'good_interval': ('-log(good interval length)', 'x for one val of tol')
			}

			x_label, y_label = label_dict.get(key)
			print (x_label, y_label)


			x.add_subplot(self.plot_one(xs,ys, x_label, y_label, function_name))
		plt.show()
		self.i  =1

	def plot_all_functions(self):

		names = ('sqrd', 'sqrt','fourth',)# 'potential')

		for name, data in zip(names, res):

			self.plot_one_function(name, data)

if __name__ == '__main__':
	run = Example()
	res = run.do_all_tests()
	plot = Plot()
	plot.plot_all_functions()

