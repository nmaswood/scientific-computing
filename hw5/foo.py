def method(INPUT_INDEX):

	fib = {0:0, 1:1}

	for index in range(2,INPUT_INDEX + 1):
		fib[index] = fib[index-1] + fib[index -2]

	return fib[INPUT_INDEX]


res = method(7)

print (res)

	