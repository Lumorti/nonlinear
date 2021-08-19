CXX=g++
CXXFLAGS=-Wfatal-errors -O3
DEBUGFLAGS=-Wfatal-errors -O0 -pg -g
LIBS=-I/usr/include/eigen3

all:
	$(CXX) $(CXXFLAGS) -o nl main.cpp $(LIBS)

debug:
	$(CXX) $(DEBUGFLAGS) -o nl main.cpp $(LIBS)
