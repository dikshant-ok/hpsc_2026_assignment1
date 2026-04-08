CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall

SRCS = main.cpp init.cpp forces.cpp integrator.cpp diagnostics.cpp neighbour.cpp
OBJS = $(SRCS:.cpp=.o)

# Default: serial build
all: dem_serial dem_parallel

dem_serial: $(SRCS)
	$(CXX) $(CXXFLAGS) -o dem_serial $(SRCS)

dem_parallel: $(SRCS)
	$(CXX) $(CXXFLAGS) -fopenmp -D_OPENMP -o dem_parallel $(SRCS)

# Run all tests and simulations
run_tests: dem_serial
	./dem_serial tests

run_serial: dem_serial
	./dem_serial serial

run_parallel: dem_parallel
	./dem_parallel parallel

run_neighbour: dem_serial
	./dem_serial neighbour

run_all: dem_serial dem_parallel
	./dem_serial tests
	./dem_serial serial
	./dem_parallel parallel
	./dem_serial neighbour

clean:
	rm -f dem_serial dem_parallel *.o *.csv

.PHONY: all clean run_tests run_serial run_parallel run_neighbour run_all
