CC = gcc
CFLAGS = -O2 -fPIC -pedantic-errors -Wall -Wextra -Wsign-conversion -Wconversion -Werror -std=c99 -fopenmp
LIBS = -lgsl -lgslcblas -lm
GSL = /home/shadrin/github/gsl/install

all: mostlib.so

mostlib.so: mostlib.o
	$(CC) $(CFLAGS) -shared -L$(GSL)/lib -o mostlib.so mostlib.o $(LIBS)

mostlib.o:
	$(CC) $(CFLAGS) -I$(GSL)/include -c mostlib.c

clean:
	rm *.o
