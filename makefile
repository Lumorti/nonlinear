CXX=g++
CXXFLAGS=-Wfatal-errors -O3
LIBS=-I/usr/include/eigen3

all:
	$(CXX) $(CXXFLAGS) -o nl main.cpp $(LIBS)
