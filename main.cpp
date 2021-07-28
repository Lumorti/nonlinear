#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <iomanip>
#include <math.h>

// Allow use of "2i" for complex
using namespace std::complex_literals;

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/../unsupported/Eigen/KroneckerProduct>

// Objective function
double f(Eigen::VectorXd<double> x) {


}

// Constraint function
Eigen::VectorXd<double> g(Eigen::VectorXd<double> x) {


}

// Function turning x to X
Eigen::MatrixXd<double> X(Eigen::VectorXd<double> x) {


}

// Standard cpp entry point
int main (int argc, char ** argv) {

	// Defining the MUB problem
	int d = 2;
	int sets = 2;

	// Useful quantities
	int numPerm = sets*(sets-1)/2;
	int numMeasureB = sets;
	int numOutcomeB = d;
	int numUnique = (d*(d+1)) / 2;

	// Sizes of matrices
	int n = numMeasureB*numOutcomeB*numUnique;
	int m = 0;
	int p = numMeasureB*numOutcomeB*d;

	// An interior point has three components
	struct interiorPoint {
		Eigen::VectorXd<double> x(n);
		Eigen::VectorXd<double> y(m);
		Eigen::MatrixXd<double> Z(p, p);
	}

	// Optimisation parameters
	double epsilon = 1e-5;
	double M_c = 0.1;
	
	// The interior point to optimise
	interiorPoint w;

	// Outer loop
	int k = 0;
	while (false) {


	}

	// Everything went fine
	return 0;

}

