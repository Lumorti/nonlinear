CXX=g++
CXXFLAGS=-Wfatal-errors -O3
DEBUGFLAGS=-Wfatal-errors -O3 -pg -g
LIBS=-I/usr/include/eigen3
HDIR=${MSKHOME}/mosek/9.2/tools/platform/linux64x86/h
LIBDIR=${MSKHOME}/mosek/9.2/tools/platform/linux64x86/bin
LDLIBSMOSEK= -I$(HDIR) -L$(LIBDIR) -Wl,-rpath-link,$(LIBDIR) -Wl,-rpath=$(LIBDIR) -lmosek64 -lfusion64 

all:
	$(CXX) $(CXXFLAGS) -o nl main.cpp $(LIBS) $(LDLIBSMOSEK)

debug:
	$(CXX) $(DEBUGFLAGS) -o nl main.cpp $(LIBS)

old:
	$(CXX) $(CXXFLAGS) -fopenmp -o old old.cpp $(LIBS)
