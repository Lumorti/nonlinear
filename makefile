CXX=g++
CXXFLAGS=-Wfatal-errors -O3
DEBUGFLAGS=-Wfatal-errors -O3 -pg -g
LIBS=-I/usr/include/eigen3 -fopenmp

all:
	$(CXX) $(CXXFLAGS) -o nonlin src/main.cpp $(LIBS)

debug:
	$(CXX) $(DEBUGFLAGS) -o nonlin src/main.cpp $(LIBS)

