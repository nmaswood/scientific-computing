import matplotlib.pyplot as plt
from math import log, pi, sin

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

POTENTIAL = Function(lambda x: 1 if x < 0.5 else 0, [0,1])

# Result
#
# Float -> Int -> Int-> Result
# A tuple of the approximation of integral area
# and the deepest level taken to get there as well
# the number of function calls

class Result():

	def __init__(self, result, max_depth, function_calls):

		self.result = result
		self.max_depth = max_depth
		self.function_calls = function_calls


# AdaptiveQuadrature
#
# Implements the adapative quadrature algorithim

class AdaptiveQuadrature():

	def __init__(self):

		# Keeps track of the number function calls
		self.function_calls = 0

		# Keeps track of maximum dept
		self.max_depth = float('-inf')

	def adaptive_quadrature(f, tolerance_original, a_original,b_original, max_level, special = False):
		"""

		adaptive_quadrature

		function -> Float -> Float -> Float -> Int -> Bool -> Result

		Calculates the adaptive quadrative for a function
		over an interval. If the special flag is set returns
		instead of the fine approximation and combination of the
		fine and coarse for a better estimate.

		"""

		def sub_routine(tolerance, a, b, level):

			f_a, f_b = f(a), f(b)

			I_1 = ((b-a)/2) * (f_a + f_b)

			m = (a+b) / 2

			f_m = f(m)

			I_2 = ((b-a)/4) * (f_a + 2 * f_m + f_b)

			if abs(I_1 - I_2) < (3 * (b-a) * tolerance) and level >= max_level:

				self.function_calls += 1
				self.max_depth = max(self.max_depth, level)

				if special:
					return  (4 * I_2  - I_1) / 3

				return I_2

			else:

				return  sub_routine(tolerance / 2, a, m, level + 1) + \
				        sub_routine(tolerance / 2, m, b, level + 1)

		approximation = sub_routine(tolerance_original, a_original,b_original, 0)

		return Result(approximation, self.max_depth, self.function_calls)

# Example
#
# Holds the methods and constants
# used to test and the different
# factors against one another.

class Example(AdaptiveQuadrature):

	def __init__(self):

		self.MAX_LEVEL = 30
		self.PADDING = 1e-30
		self.ACTUAL_ERROR_TOLERANCE = 1e-30
		self.TOLERANCE_RANGE = (1e-2, 3e-3, 1e-3, 3e-4, 3e-5, 1e-5, 3e-6, 1e-6)

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
				self.adaptive_quadrature(f, tol_variable, a,b, self.MAX_LEVEL) for 
				tol_variable in self.TOLERANCE_RANGE
			]

			xs = [x.function_calls for x in results]

			ys = [log(1/tol_variable + self.PADDING) for tol_variable in TOLERANCE_RANGE]

			return (xs,ys)

		def actual_error_test(f,a,b):

			"""

			actual error test

			function -> Float -> Float -> Tuple <Float,Float>

			Returns a tuple where xs is ____
			and ys is _____

			"""

			actual_error = self.actual_quadrature(f,ACTUAL_ERROR_TOLERANCE, a,b, self.MAX_LEVEL)

			xs = [1/ log(actual_error)] * len(TOLERANCE_RANGE)

			ys = []

		def good_interval_length_test(f,a,b):

			"""

			good_interval_length_test

			function -> Float -> Float -> Tuple <Float,Float>

			Returns a tuple where xs is _
			and ys is __

			"""


			pass

		return {
			'function_evaluations': function_evaluations_test(f,a,b),
			'actual_error': actual_error_test(f,a,b),
			'good_interval': good_interval_length_test(f,a,b)
		}

	def do_all_tests():

		"""
		do_all_tests

		-> Dict< String, Dict<String,Tuple<List Float, List Float>>

		"""

		results = []
		
		for function_struct in (SQRD, SQRT, FOURTH, POTENTIAL):

			f = function_struct.function
			a,b = function_struct.interval

			results.append(self.calculate(f,a,b))

		return results

class Plot():
	
	"""

	plot_one

	Float -> Float -> xlabel -> ylabel

	Creates a single plot

	"""

	def plot_one(self,xs, ys, xlabel, ylabel):

		plt.plot(xs, ys)
		plt.xlabel = xlabel
		plt.ylabel = ylabel
		plt.show()

if __name__ == '__main__':
	print ("hello world")
