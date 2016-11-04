
a = '01-21-1995'

class Date():

	def __init__(self,raw_str):

		day, month, year = raw_str.split('-',2)

		self.day = day
		self.month = month
		self.year = year

		self.m_31 = set([1,3, 5, 7, 8,10, 12])

	def _absolute_days(self):

		return self.day + month_to_days() + year_to_days()

	def _month_to_days(self,month):

		if month == 2:
			return 28
		elif month in self.m_31:
			return 31
		else:
			return 30

	def _months_to_days(self, month):

		return sum(self.month_to_days(i) for i in range(month))

	def _is_leap_year(self, year):

		return (
			year%4==0 and 
			(year%100!=0 or year%400==0)
			)

	def _years_to_days(self, year):

		acc = 0

		yearPrime = year

		i = 4

		while i and yearPrime:

			yearPrimePrime = yearPrime - 1

			if self._is_leap_year(yearPrimePrime):
				break
			else:
				acc += 365

			yearPrime -= 1

		if yearPrime == 0:
			return acc

		leapYears = yearPrime - int(yearPrime / 4)

		nonLeapYears = yearPrime - leapYears

		return acc + (366  * leapYears) + (365 * nonLeapYears)

	def __sub__(self, other):

		return abs(self.absolute_days() - other.absolute_days())









