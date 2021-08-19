CXX=g++
CXXFLAGS=-Wfatal-errors -O3 -fopenmp
DEBUGFLAGS=-Wfatal-errors -O3 -pg -g
LIBS=-I/usr/include/eigen3

all:
	$(CXX) $(CXXFLAGS) -o nl main.cpp $(LIBS)

debug:
	$(CXX) $(DEBUGFLAGS) -o nl main.cpp $(LIBS)
