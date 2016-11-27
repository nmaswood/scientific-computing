LIB = -lm
CPP = g++
CPPFLAGS = -O -Wall
#CPPFLAGS = -g -Wall

OBJS = \
conjugateGrad.o\
main.o\
matrix.o\
utility.o


runit : tryit
	tryit

tryit : $(OBJS)
	$(CPP) $(CPPFLAGS) -o $@ $(OBJS) $(LIB)

conjugateGrad.o : conjugateGrad.cpp conjugateGrad.h matrix.h utility.h
	$(CPP) -c $(CPPFLAGS) conjugateGrad.cpp

main.o : main.cpp conjugateGrad.h matrix.h utility.h
	$(CPP) -c $(CPPFLAGS) main.cpp

matrix.o : matrix.cpp matrix.h utility.h
	$(CPP) -c $(CPPFLAGS) matrix.cpp

utility.o : utility.cpp utility.h
	$(CPP) -c $(CPPFLAGS) utility.cpp

spotless : 
	rm tryit *~  *.o  example*
