#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <chrono>
#include <iomanip>
#include <math.h>

// Allow use of "2i" for complex
using namespace std::complex_literals;

// For openmp 
#include <omp.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/../unsupported/Eigen/KroneckerProduct>
#include <Eigen/../unsupported/Eigen/MatrixFunctions>

// Defining the MUB problem
int d = 2;
int sets = 2;
std::string initMode = "fixed";

// Optimisation parameters
double extraDiag = 1.0;
long int maxOuterIter =  1000000000;
long int maxInnerIter =  1000000000;
long int maxTotalInner = 1000000000;
double muScaling = 10;
double gThresh = 1e-5;
int numCores = 1;
bool useBFGS = false;
double BFGSmaxG = 1e-10;

// Parameters between 0 and 1
double gammaVal = 0.1;
double epsilonZero = 0.9;
double beta = 0.1;

// Parameters greater than 0
double epsilon = 1e-5; 
double M_c = 1e10;
double mu = 1.0;
double nu = 0.9;
double rho = 0.5;

// Useful quantities to define later
int numPerm = 0;
int numMeasureB = 0;
int numOutcomeB = 0;
int numUniquePer = 0;
int numRealPer = 0;
int numImagPer = 0;
int numMats = 0;
int numRhoMats = 0;
int numBMats = 0;
std::vector<Eigen::SparseMatrix<double>> As;
Eigen::SparseMatrix<double> C;
Eigen::VectorXd b;
Eigen::SparseMatrix<double> Q;
Eigen::SparseMatrix<double> identityp;
Eigen::SparseMatrix<double> identityn;
Eigen::VectorXd factors0;
double rt2 = std::sqrt(2);

// Sizes of matrices
int n = 0;
int m = 0;
int p = 0;
int halfP = p / 2;

// For printing
int precision = 5;
std::string outputMode = "";
int x1 = 0;
int x2 = 1;

// Pretty print a generic 1D vector
template <typename type> void prettyPrint(std::string pre, std::vector<type> arr) {

	// Used fixed precision
	std::cout << std::fixed << std::showpos << std::setprecision(precision);

	// For the first line, add the pre text
	std::cout << pre << " { ";

	// For the x values, combine them all on one line
	for (int x=0; x<arr.size(); x++) {
		std::cout << arr[x];
		if (x < arr.size()-1) {
			std::cout << ", ";
		}
	}

	// Output the row
	std::cout << "}" << std::endl;

	// Reset things for normal output
	std::cout << std::noshowpos;

}

// Pretty print a 1D dense Eigen array
template <typename type>
void prettyPrint(std::string pre, Eigen::Matrix<type, -1, 1> arr) {

	// Used fixed precision
	std::cout << std::fixed << std::showpos << std::setprecision(precision);

	// Spacing
	std::cout << pre << " { ";

	// For the x values, combine them all on one line
	for (int x=0; x<arr.rows(); x++) {
		std::cout << std::setw(5) << arr(x);
		if (x < arr.rows()-1) {
			std::cout << ", ";
		}
	}

	// Output the row
	std::cout << "}" << std::endl;

	// Reset things for normal output
	std::cout << std::noshowpos;

}
// Pretty print a general 2D dense Eigen array
template <typename type>
void prettyPrint(std::string pre, Eigen::Matrix<type, -1, -1> arr) {

	// Used fixed precision
	std::cout << std::fixed << std::showpos << std::setprecision(precision);

	// Loop over the array
	std::string rowText;
	for (int y=0; y<arr.rows(); y++) {

		// For the first line, add the pre text
		if (y == 0) {
			rowText = pre + "{";

		// Otherwise pad accordingly
		} else {
			rowText = "";
			while (rowText.length() < pre.length()+1) {
				rowText += " ";
			}
		}

		// Spacing
		std::cout << rowText << " { ";

		// For the x values, combine them all on one line
		for (int x=0; x<arr.cols(); x++) {
			std::cout << std::setw(precision+2) << arr(y,x);
			if (x < arr.cols()-1) {
				std::cout << ", ";
			}
		}

		// Output the row
		std::cout << "}";
		if (y < arr.rows()-1) {
			std::cout << ",";
		} else {
			std::cout << " } ";
		}
		std::cout << std::endl;

	}

	// Reset things for normal output
	std::cout << std::noshowpos;

}

// Pretty print a general 2D sparse Eigen array
template <typename type>
void prettyPrint(std::string pre, Eigen::SparseMatrix<type> arr) {

	// Extract the dense array and then call the routine as normal
	prettyPrint(pre, Eigen::Matrix<type,-1,-1>(arr));

}

// Pretty print a double
void prettyPrint(std::string pre, double val) {
	std::cout << pre << val << std::endl;
}

// Calculate a "norm" for a 3D "matrix"
double norm3D(std::vector<Eigen::MatrixXd> M) {
	double val = 0;
	for (int i=0; i<M.size(); i++) {
		val += M[i].norm();
	}
	return val;
}

// Function turning X to x
Eigen::VectorXd matToVec(Eigen::SparseMatrix<double> X) {

	// Create a blank n vector
	Eigen::VectorXd newVec(n);

	// Extract each element
	for (int i=0; i<n; i++) {
		newVec(i) = X.cwiseProduct(As[i]).sum()/2;
	}

	// Return this new vector
	return newVec;

}

// Function turning x to X
Eigen::SparseMatrix<double> vecToMat(Eigen::VectorXd x) {

	// Create a blank p by p matrix
	Eigen::SparseMatrix<double> newMat(p, p);

	// Extract each element
	for (int i=0; i<n; i++) {
		newMat += As[i]*x(i);
	}

	// Return this new matrix
	return newMat;

}

// Objective function
double f(Eigen::VectorXd x) {
	return -((x.adjoint()*Q*x)(0));
}

// Derivative of the objective function
Eigen::VectorXd delf(Eigen::VectorXd x) {
	return -2*x.adjoint()*Q;
}

// Second derivative of the objective function
Eigen::MatrixXd del2f(Eigen::VectorXd x) {
	return -2*Q;
}
				
// Constraint function
Eigen::VectorXd g(Eigen::VectorXd x) {

	// Create an empty vector
	Eigen::VectorXd returnVec(m);

	// Need numMats to be in "vector" form (size 1 vector)
	Eigen::VectorXd N = Eigen::VectorXd::Constant(1, numMats);

	// First element is the x^Tx = N constraint
	returnVec.head(1) = x.adjoint()*x - N;

	// Last m-1 elements are the normalisation constraints
	returnVec.tail(m-1) = C*x - b;

	// Return the vector of constraints
	return returnVec;

}

// Derivative of the constraint function
Eigen::MatrixXd delg(Eigen::VectorXd x) {

	// Create an empty matrix
	Eigen::MatrixXd returnMat(m, n);

	// First element is the x^Tx = N constraint
	returnMat.block(0,0,1,n) = 2*x.transpose();

	// Last m-1 elements are the normalisation constraints
	returnMat.block(1,0,m-1,n) = C;

	// Return the vector of constraints
	return returnMat;

}

// Second derivatives of the constraint function
std::vector<Eigen::MatrixXd> del2g(Eigen::VectorXd x) {

	// Create an empty 3D matrix (vector of matrices)
	std::vector<Eigen::MatrixXd> returnMat(n, Eigen::MatrixXd::Zero(n, m));

	// The x^Tx = N constraint
	for (int i=0; i<n; i++) {
		returnMat[i].col(0) = Eigen::MatrixXd::Constant(n, 1, 2);
	}

	// Return the vector of constraints
	return returnMat;

}

// The dual
double dual(Eigen::VectorXd y, Eigen::SparseMatrix<double> Z) {
	Eigen::VectorXd z = matToVec(Z);
	Eigen::VectorXd res = -0.5 * (y.head(m-1).adjoint()*C - z.adjoint()) * (Q + y.tail(1)*identityp).adjoint() * (C*y.head(m-1) - z) - y.tail(1)*numMats - y.head(m-1).adjoint()*b;
	return std::real(res(0));
}

// The Lagrangian 
double L(Eigen::VectorXd x, Eigen::SparseMatrix<double> X, Eigen::VectorXd y, Eigen::SparseMatrix<double> Z) {
	return std::real(f(x) - y.dot(g(x)) - X.cwiseProduct(Z).sum());
}

// Differential of the Lagrangian given individual components
Eigen::VectorXd delL(Eigen::VectorXd y, Eigen::SparseMatrix<double> Z, Eigen::VectorXd delfCached, Eigen::MatrixXd A_0) {

	// Calculate A* Z
	Eigen::VectorXd AStarZ = matToVec(Z);

	// Return this vector
	return delfCached - A_0.transpose()*y - AStarZ;

}

// Double differential of the Lagrangian given an interior point
Eigen::MatrixXd del2L(Eigen::VectorXd x, Eigen::VectorXd y) {

	Eigen::MatrixXd prod(n, n);
	std::vector<Eigen::MatrixXd> del2gCached = del2g(x);
	for (int i=0; i<n; i++) {
		prod.row(i) = del2gCached[i] * y; 
	}

	// In our case the second derivative of the A dot Z is zero
	return del2f(x) - prod;

}

// Function giving the norm of a point, modified by some mu
double rMag(double mu, Eigen::SparseMatrix<double> Z, Eigen::SparseMatrix<double> X, Eigen::VectorXd delLCached, Eigen::VectorXd gCached) {

	// The left part of the square root
	Eigen::VectorXd left = Eigen::VectorXd::Zero(n+m);
	left << delLCached, gCached;

	// The right part of the square root
	Eigen::MatrixXd right = X*Z - mu*identityp;

	// Sum the l^2/Frobenius norms
	double val = std::sqrt(left.squaredNorm() + right.squaredNorm());

	// Return this magnitude
	return val;

}

// The merit function
double F(Eigen::VectorXd x, Eigen::SparseMatrix<double> Z, double mu) {

	// Cache the X matrix
	Eigen::SparseMatrix<double> X = vecToMat(x);
	Eigen::VectorXd gCached = g(x);
	double XDeter = std::real(Eigen::MatrixXd(X).determinant());
	double ZDeter = std::real(Eigen::MatrixXd(Z).determinant());

	// Calculate the two components
	double FBP = std::real(f(x) - mu*std::log(XDeter) + rho*gCached.norm());
	double FPD = std::real(X.cwiseProduct(Z).sum() - mu*std::log(XDeter*ZDeter));

	// Return the sum
	return FBP + nu*FPD;

}

// The change in merit function
double deltaF(Eigen::MatrixXd deltaZ, Eigen::SparseMatrix<double> ZInverse, Eigen::SparseMatrix<double> Z, Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> XInverse, Eigen::VectorXd delfCached, Eigen::VectorXd gCached, Eigen::MatrixXd A_0, Eigen::VectorXd deltax) {

	// Calculate the deltaX matrix
	Eigen::MatrixXd deltaX = vecToMat(deltax);

	// Calculate the two components
	double FBP = std::real(delfCached.dot(deltax) - mu*(XInverse*deltaX).trace() + rho*((gCached+A_0*deltax).norm()-gCached.norm()));
	double FPD = std::real((deltaX*Z + X*deltaZ - mu*XInverse*deltaX - mu*ZInverse*deltaZ).trace());

	// Return the sum
	return FBP + nu*FPD;

}

// Returns true if a matrix can be Cholesky decomposed
bool isPD(Eigen::MatrixXd G) {
	return G.llt().info() != Eigen::NumericalIssue;
}
bool isPSD(Eigen::MatrixXd G) {
	return (G+(1e-13)*Eigen::MatrixXd::Identity(G.cols(), G.rows())).llt().info() != Eigen::NumericalIssue;
}

// Given a matrix, make it be positive definite
void makePD(Eigen::MatrixXd G) {

	// See if G is already PD
	if (!isPD(G)) {

		// If G+sigma*I is PD
		double sigma = 1;
		if (isPD(G + sigma*identityn)) {

			// Decrease sigma until it isn't
			while (isPD(G + sigma*identityn)) {
				sigma /= 2;
			}

			// Then return to the one that was still PD
			sigma *= 2;

		// If G+sigma*I is not PD
		} else {

			// Increase sigma until it is
			while (!isPD(G + sigma*identityn)) {
				sigma *= 2;
			}

		}

		// Update this new G
		G = G + sigma*identityn;

	}

}

// Standard cpp entry point
int main(int argc, char ** argv) {

	// Loop over the command-line arguments
	for (int i=1; i<argc; i++){

		// Convert the char array to a standard string for easier processing
		std::string arg = argv[i];

		// If asking for help
		if (arg == "-h" || arg == "--help") {
			std::cout << "" << std::endl;
			std::cout << "---------------------------------" << std::endl;
			std::cout << "  Program that checks for MUBs" << std::endl;
			std::cout << "  using a non-linear SDP solver" << std::endl;
			std::cout << "---------------------------------" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "       main options          " << std::endl;
			std::cout << " -h               show the help" << std::endl;
			std::cout << " -d [int]         set the dimension" << std::endl;
			std::cout << " -n [int]         set the number of measurements" << std::endl;
			std::cout << " -c [int]         set how many cores to use" << std::endl;
			std::cout << " -V [int] [int]   visualise the search space" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "       output options          " << std::endl;
			std::cout << " -p [int]         set the precision" << std::endl;
			std::cout << " -B               output only the iter count for benchmarking" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "       init options          " << std::endl;
			std::cout << " -R               use a random seed" << std::endl;
			std::cout << " -Y               start nearby the ideal if known" << std::endl;
			std::cout << " -T [dbl]         set the threshold for the initial g(x)" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "    parameter options          " << std::endl;
			std::cout << " -S [dbl]         set max g(x) for BFGS" << std::endl;
			std::cout << " -e [dbl]         set epsilon" << std::endl;
			std::cout << " -M [dbl]         set M_c" << std::endl;
			std::cout << " -b [dbl]         set beta" << std::endl;
			std::cout << " -G [dbl]         set the factor of g(x)" << std::endl;
			std::cout << " -D [dbl]         set the extra diagonal" << std::endl;
			std::cout << " -g [dbl]         set gamma" << std::endl;
			std::cout << " -s [dbl]         set mu scaling (mu->mu/this)" << std::endl;
			std::cout << " -m [dbl]         set initial mu" << std::endl;
			std::cout << " -E [dbl]         set epsilon_0" << std::endl;
			std::cout << " -F [dbl]         set the factor of f(x)" << std::endl;
			std::cout << " -r [dbl]         set rho" << std::endl;
			std::cout << " -N [dbl]         set nu" << std::endl;
			std::cout << "                           " << std::endl;
			std::cout << "      limit options          " << std::endl;
			std::cout << " -I [int]         set outer iteration limit" << std::endl;
			std::cout << " -i [int]         set inner iteration limit" << std::endl;
			std::cout << " -t [int]         set total inner iteration limit" << std::endl;
			std::cout << "" << std::endl;
			return 0;

		// Set the number of measurements 
		} else if (arg == "-d") {
			d = std::stoi(argv[i+1]);
			i += 1;

		// Use the BFGS update method
		} else if (arg == "-S") {
			useBFGS = true;
			BFGSmaxG = std::stod(argv[i+1]);
			i += 1;

		// Set the number of cores 
		} else if (arg == "-c") {
			numCores = std::stoi(argv[i+1]);
			i += 1;

		// Set the max total inner iteration
		} else if (arg == "-t") {
			maxTotalInner = std::stoi(argv[i+1]);
			i += 1;

		// If told to only output the iteration count
		} else if (arg == "-B") {
			outputMode = "B";

		// If told to use a random start rather than the default seed
		} else if (arg == "-R") {
			initMode = "random";

		// Visualise part of the search space
		} else if (arg == "-V") {
			initMode = "vis";
			outputMode = "vis";
			x1 = std::stoi(argv[i+1]);
			x2 = std::stoi(argv[i+2]);
			i += 2;

		// If told to use near the known exact
		} else if (arg == "-Y") {
			initMode = "nearby";

		// Set the number of sets
		} else if (arg == "-n") {
			sets = std::stoi(argv[i+1]);
			i += 1;

		// Set the outer iteration limit
		} else if (arg == "-I") {
			maxOuterIter = std::stoi(argv[i+1]);
			i += 1;

		// Set the inner iteration limit
		} else if (arg == "-i") {
			maxInnerIter = std::stoi(argv[i+1]);
			i += 1;

		// Set the threshold for the initial g(x)
		} else if (arg == "-T") {
			gThresh = std::stod(argv[i+1]);
			i += 1;

		// Set the mu scaling
		} else if (arg == "-s") {
			muScaling = std::stod(argv[i+1]);
			i += 1;

		// Set the extra diagonal 
		} else if (arg == "-D") {
			extraDiag = std::stod(argv[i+1]);
			i += 1;

		// Set the gamma
		} else if (arg == "-g") {
			gammaVal = std::stod(argv[i+1]);
			i += 1;

		// Set rho
		} else if (arg == "-r") {
			rho = std::stod(argv[i+1]);
			i += 1;

		// Set the precision
		} else if (arg == "-p") {
			precision = std::stoi(argv[i+1]);
			i += 1;

		// Set epsilon
		} else if (arg == "-e") {
			epsilon = std::stod(argv[i+1]);
			i += 1;

		// Set epsilon zero
		} else if (arg == "-E") {
			epsilonZero = std::stod(argv[i+1]);
			i += 1;

		// Set M_c
		} else if (arg == "-M") {
			M_c = std::stod(argv[i+1]);
			i += 1;

		// Set beta
		} else if (arg == "-b") {
			beta = std::stod(argv[i+1]);
			i += 1;

		// Set nu
		} else if (arg == "-N") {
			nu = std::stod(argv[i+1]);
			i += 1;

		// Set mu
		} else if (arg == "-m") {
			mu = std::stod(argv[i+1]);
			i += 1;

		}

	}

	// Tell openmp the number of cores to use
	omp_set_num_threads(numCores);

	// Output formatting
	std::cout << std::setprecision(precision);

	// Start the timer 
	auto t1 = std::chrono::high_resolution_clock::now();

	// Useful quantities
	numPerm = sets*(sets-1)/2;
	numMeasureB = sets;
	numOutcomeB = d;
	numRealPer = (d*(d+1))/2;
	numImagPer = (d*(d+1))/2-d;
	numUniquePer = numRealPer + numImagPer;
	numRhoMats = numPerm*numOutcomeB*numOutcomeB;
	numBMats = numMeasureB*numOutcomeB;
	numMats = numRhoMats + numBMats;

	// Sizes of matrices
	n = numMats*numUniquePer;
	m = 1 + numMeasureB*numUniquePer + numMats;
	p = numMats*d*2;
	halfP = p / 2;

	// Cache an identity matrix
	identityp = Eigen::MatrixXd::Identity(p, p).sparseView();
	identityn = Eigen::MatrixXd::Identity(n, n).sparseView();

	// A vector which has 0 on the terms which will become off-diagonals
	factors0 = Eigen::VectorXd::Constant(numUniquePer, 0);
	for (int j=0; j<d; j++) {
		factors0(j*(2*d-j+1)/2) = 1;
	}

	// Output various bits of info about the problem/parameters
	if (outputMode == "") {
		std::cout << "--------------------------------" << std::endl;
		std::cout << "          System Info           " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "               d = " << d << std::endl;
		std::cout << "            sets = " << sets << std::endl;
		std::cout << "        num rhos = " << numRhoMats << std::endl;
		std::cout << "          num Bs = " << numBMats << std::endl;
		std::cout << "    vals per mat = " << numUniquePer << std::endl;
		std::cout << "  size of vector = " << n << " ~ " << n*16 / (1024*1024) << " MB " << std::endl;
		std::cout << "  size of matrix = " << p << " x " << p << " ~ " << p*p*16 / (1024*1024) << " MB " << std::endl;
		std::cout << "" << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "        Parameters Used           " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "         epsilon = " << epsilon << std::endl;
		std::cout << "       epsilon_0 = " << epsilonZero << std::endl;
		std::cout << "     g(x) thresh = " << gThresh << std::endl;
		std::cout << "      initial mu = " << mu << std::endl;
		std::cout << "             M_c = " << M_c << std::endl;
		std::cout << "  extra diagonal = " << extraDiag << std::endl;
		std::cout << "       muScaling = " << muScaling << std::endl;
		std::cout << "             rho = " << rho << std::endl;
		std::cout << "              nu = " << nu << std::endl;
		std::cout << "           gamma = " << gamma << std::endl;
		std::cout << "            beta = " << beta << std::endl;
		std::cout << "           cores = " << numCores << std::endl;
		std::cout << "" << std::endl;
	}

	// The "ideal" value
	double maxVal = d*d*numPerm*(1+1/std::sqrt(d));

	// Calculate the Q matrix defining the objective
	Q = Eigen::SparseMatrix<double>(n, n);
	std::vector<Eigen::Triplet<double>> tripsQ;

	// For each pairing of B matrices
	int vertLoc = 0;
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// The indices of the matrices
					int horizLoc1 = (numRhoMats + i*numOutcomeB + k)*numUniquePer;
					int horizLoc2 = (numRhoMats + j*numOutcomeB + l)*numUniquePer;

					// Loop over the real elements per vector section
					for (int m=0; m<numRealPer; m++) {
						tripsQ.push_back(Eigen::Triplet<double>(vertLoc+m, horizLoc1+m, 1));
						tripsQ.push_back(Eigen::Triplet<double>(vertLoc+m, horizLoc2+m, 1));
						tripsQ.push_back(Eigen::Triplet<double>(horizLoc1+m, vertLoc+m, 1));
						tripsQ.push_back(Eigen::Triplet<double>(horizLoc2+m, vertLoc+m, 1));
					}

					// Loop over the real elements per vector section
					for (int m=numRealPer; m<numUniquePer; m++) {
						tripsQ.push_back(Eigen::Triplet<double>(vertLoc+m, horizLoc1+m, -1));
						tripsQ.push_back(Eigen::Triplet<double>(vertLoc+m, horizLoc2+m, -1));
						tripsQ.push_back(Eigen::Triplet<double>(horizLoc1+m, vertLoc+m, -1));
						tripsQ.push_back(Eigen::Triplet<double>(horizLoc2+m, vertLoc+m, -1));
					}

					// Next section
					vertLoc += numUniquePer;

				}
			}
		}
	}

	// Construct Q from these triplets
	Q.setFromTriplets(tripsQ.begin(), tripsQ.end());
	
	// Calculate the C matrix defining the normalisation constraints
	C = Eigen::SparseMatrix<double>(m-1, n);
	std::vector<Eigen::Triplet<double>> tripsC;

	// The first part are the sum to identity constraints
	vertLoc = 0;
	for (int i=0; i<numMeasureB; i++) {
		for (int k=0; k<numOutcomeB; k++) {

			// The index of the matrix
			int horizLoc1 = (numRhoMats + i*numOutcomeB + k)*numUniquePer;

			// Loop over the elements per vector section
			for (int l=0; l<numUniquePer; l++) {
				tripsC.push_back(Eigen::Triplet<double>(vertLoc+l, horizLoc1+l, 1));
			}

		}

		// Next section
		vertLoc += numUniquePer;

	}
	
	// The second part are the trace one constraints
	for (int i=0; i<numMats; i++) {

		// The index of the matrix
		int horizLoc1 = i*numUniquePer;

		// Loop over the elements per vector section
		for (int l=0; l<numUniquePer; l++) {
			tripsC.push_back(Eigen::Triplet<double>(vertLoc, horizLoc1+l, factors0(l)));
		}

		// Next section
		vertLoc += 1;

	}
					
	// Construct C from these triplets
	C.setFromTriplets(tripsC.begin(), tripsC.end());
	
	// Calculate the b vector defining the normalisation constraints
	b = Eigen::VectorXd::Ones(m-1);
	for (int i=0; i<numMeasureB*numUniquePer; i++) {
		b(i) = factors0(i % numUniquePer);
	}

	// Calculate the A matrices for turning x -> X
	As = std::vector<Eigen::SparseMatrix<double>>(n);
	double oneOverRt2 = 1.0 / std::sqrt(2);
	int matx = 0;
	int maty = 0;
	for (int i=0; i<n; i++) {

		// For each A
		Eigen::SparseMatrix<double> A(p, p);
		std::vector<Eigen::Triplet<double>> tripsA;

		// The location of the matrix for this element
		int matInd = std::floor(i / numUniquePer);
		int matLoc = matInd*d*2;

		// Reset the pos at the start of each new matrix
		int reli = i % numUniquePer;
		if (reli == 0) {
			matx = 0;
			maty = 0;
		} else if (reli == numRealPer) {
			matx = 1;
			maty = 0;
		}

		// If it's a non-diagonal
		if (matx != maty) {

			// If it's a real component
			if (reli < numRealPer) {
				tripsA.push_back(Eigen::Triplet<double>(matLoc+matx, matLoc+maty, oneOverRt2));
				tripsA.push_back(Eigen::Triplet<double>(matLoc+maty, matLoc+matx, oneOverRt2));
				tripsA.push_back(Eigen::Triplet<double>(matLoc+matx+d, matLoc+maty+d, oneOverRt2));
				tripsA.push_back(Eigen::Triplet<double>(matLoc+maty+d, matLoc+matx+d, oneOverRt2));

			// If it's imaginary
			} else {
				tripsA.push_back(Eigen::Triplet<double>(matLoc+matx+d, matLoc+maty, oneOverRt2));
				tripsA.push_back(Eigen::Triplet<double>(matLoc+maty+d, matLoc+matx, -oneOverRt2));
				tripsA.push_back(Eigen::Triplet<double>(matLoc+matx, matLoc+maty+d, -oneOverRt2));
				tripsA.push_back(Eigen::Triplet<double>(matLoc+maty, matLoc+matx+d, oneOverRt2));

			}

		// If it's a non-diagonal
		} else {
			tripsA.push_back(Eigen::Triplet<double>(matLoc+matx, matLoc+maty, 1));
			tripsA.push_back(Eigen::Triplet<double>(matLoc+matx+d, matLoc+maty+d, 1));

		}

		// Update the position location
		matx++;
		if (matx >= d) {
			if (reli < numRealPer) {
				maty++;
				matx = maty;
			} else {
				maty++;
				matx = maty+1;
			}
		}

		// Construct this sparse matrix
		A.setFromTriplets(tripsA.begin(), tripsA.end());

		// Add to the list
		As[i] = A;

	}

	// The primal var (and alt forms)
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
	Eigen::SparseMatrix<double> X(p, p);
	Eigen::MatrixXd XDense = Eigen::MatrixXd::Zero(p, p);

	// The dual vars (and alt forms)
	Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
	Eigen::SparseMatrix<double> Z(p, p);
	Eigen::MatrixXd ZDense = Eigen::MatrixXd::Zero(p, p);

	// Seed so it's random each time
	if (initMode == "random") {
		srand((unsigned int) time(0));
	}

	// If starting with a random matrix (seeded or not)
	if (initMode == "random" || initMode == "fixed") {

		// Start with a bunch of projective measurements TODO

	// Use nearby the optimum
	} else if (initMode == "nearby") {

		// Allow entry as the list of matrices
		std::vector<std::vector<std::vector<std::complex<double>>>> Ms(numMeasureB*numOutcomeB);

		// From the seesaw
		if (d == 2 && sets == 2) {
			Ms[0] = { { 1.0, 0.0 }, 
					  { 0.0, 0.0 } };
			Ms[1] = { { 0.0, 0.0 }, 
					  { 0.0, 1.0 } };
			Ms[2] = { { 0.5, 0.5 }, 
					  { 0.5, 0.5 } };
			Ms[3] = { { 0.5,-0.5 }, 
					  {-0.5, 0.5 } };
		} else if (d == 2 && sets == 3) {
			Ms[0] = { { 1.0, 0.0 }, 
					  { 0.0, 0.0 } };
			Ms[1] = { { 0.0, 0.0 }, 
					  { 0.0, 1.0 } };
			Ms[2] = { { 0.5,-0.5 }, 
					  {-0.5, 0.5 } };
			Ms[3] = { { 0.5, 0.5 }, 
					  { 0.5, 0.5 } };
			Ms[4] = { {  0.5, 0.5i }, 
					  {-0.5i, 0.5 } };
			Ms[5] = { {  0.5,-0.5i }, 
					  { 0.5i, 0.5 } };
		} else if (d == 2 && sets == 4) {
			Ms[0] = { {  0.90+0.00i,  0.15+0.26i },
				      {  0.15-0.26i,  0.10+0.00i } };
		    Ms[1] = { {  0.10+0.00i, -0.15-0.26i },
				      { -0.15+0.26i,  0.90+0.00i } };
		    Ms[2] = { {  0.58+0.00i, -0.46-0.18i },
				      { -0.46+0.18i,  0.42+0.00i } };
		    Ms[3] = { {  0.42+0.00i,  0.46+0.18i },
				      {  0.46-0.18i,  0.58+0.00i } };
		    Ms[4] = { {  0.10+0.00i, -0.01+0.30i },
				      { -0.01-0.30i,  0.90+0.00i } };
		    Ms[5] = { {  0.90+0.00i,  0.01-0.30i },
				      {  0.01+0.30i,  0.10+0.00i } };
		    Ms[6] = { {  0.58+0.00i, -0.31+0.38i },
		 		      { -0.31-0.38i,  0.42+0.00i } };
		    Ms[7] = { {  0.42+0.00i,  0.31-0.38i },
				      {  0.31+0.38i,  0.58+0.00i } };
		} else if (d == 3 && sets == 5) {
		    Ms[0] = {  {  0.50+0.00i, -0.25-0.40i,  0.11-0.12i },
                       { -0.25+0.40i,  0.45+0.00i,  0.04+0.15i },
                       {  0.11+0.12i,  0.04-0.15i,  0.05+0.00i } };
		    Ms[1] = {  {  0.18+0.00i,  0.00+0.07i, -0.18+0.33i },
                       {  0.00-0.07i,  0.03+0.00i,  0.13+0.08i },
                       { -0.18-0.33i,  0.13-0.08i,  0.79+0.00i } };
		    Ms[2] = {  {  0.33+0.00i,  0.25+0.33i,  0.07-0.21i },
                       {  0.25-0.33i,  0.52+0.00i, -0.16-0.23i },
                       {  0.07+0.21i, -0.16+0.23i,  0.15+0.00i } };
		    Ms[3] = {  {  0.09+0.00i, -0.01+0.07i,  0.19-0.20i },
                       { -0.01-0.07i,  0.05+0.00i, -0.18-0.11i },
                       {  0.19+0.20i, -0.18+0.11i,  0.86+0.00i } };
		    Ms[4] = {  {  0.78+0.00i, -0.29+0.06i, -0.24+0.16i },
                       { -0.29-0.06i,  0.11+0.00i,  0.10-0.04i },
                       { -0.24-0.16i,  0.10+0.04i,  0.11+0.00i } };
		    Ms[5] = {  {  0.13+0.00i,  0.30-0.12i,  0.05+0.04i },
                       {  0.30+0.12i,  0.84+0.00i,  0.08+0.15i },
                       {  0.05-0.04i,  0.08-0.15i,  0.03+0.00i } };
		    Ms[6] = {  {  0.19+0.00i, -0.17+0.24i, -0.17-0.19i },
                       { -0.17-0.24i,  0.47+0.00i, -0.09+0.39i },
                       { -0.17+0.19i, -0.09-0.39i,  0.35+0.00i } };
		    Ms[7] = {  {  0.14+0.00i, -0.02-0.26i, -0.23+0.03i },
                       { -0.02+0.26i,  0.48+0.00i, -0.03-0.43i },
                       { -0.23-0.03i, -0.03+0.43i,  0.38+0.00i } };
		    Ms[8] = {  {  0.67+0.00i,  0.19+0.02i,  0.40+0.16i },
                       {  0.19-0.02i,  0.05+0.00i,  0.12+0.03i },
                       {  0.40-0.16i,  0.12-0.03i,  0.27+0.00i } };
		    Ms[9] = {  {  0.57+0.00i,  0.35+0.02i, -0.32+0.12i },
                       {  0.35-0.02i,  0.22+0.00i, -0.19+0.09i },
                       { -0.32-0.12i, -0.19-0.09i,  0.21+0.00i } };
		    Ms[10] = { {  0.07+0.00i,  0.02-0.15i,  0.07-0.19i },
                       {  0.02+0.15i,  0.33+0.00i,  0.44+0.08i },
                       {  0.07+0.19i,  0.44-0.08i,  0.60+0.00i } };
		    Ms[11] = { {  0.35+0.00i, -0.38+0.13i,  0.25+0.07i },
                       { -0.38-0.13i,  0.45+0.00i, -0.24-0.17i },
                       {  0.25-0.07i, -0.24+0.17i,  0.20+0.00i } };
		    Ms[12] = { {  0.18+0.00i, -0.09+0.32i,  0.02+0.19i },
                       { -0.09-0.32i,  0.61+0.00i,  0.34-0.13i },
                       {  0.02-0.19i,  0.34+0.13i,  0.21+0.00i } };
		    Ms[13] = { {  0.15+0.00i,  0.02-0.24i,  0.10+0.25i },
                       {  0.02+0.24i,  0.37+0.00i, -0.38+0.19i },
                       {  0.10-0.25i, -0.38-0.19i,  0.48+0.00i } };
		    Ms[14] = { {  0.67+0.00i,  0.07-0.08i, -0.12-0.44i },
                       {  0.07+0.08i,  0.02+0.00i,  0.04-0.06i },
                       { -0.12+0.44i,  0.04+0.06i,  0.31+0.00i } };
		} else if (d == 4 && sets == 4) {
			Ms[0] = {  {  0.39+0.00i, -0.29-0.12i,  0.16+0.26i,  0.17-0.14i },
					   { -0.29+0.12i,  0.25+0.00i, -0.20-0.14i, -0.08+0.15i },
					   {  0.16-0.26i, -0.20+0.14i,  0.24+0.00i, -0.02-0.17i },
					   {  0.17+0.14i, -0.08-0.15i, -0.02+0.17i,  0.12+0.00i } };
			Ms[1] = {  {  0.24+0.00i,  0.27+0.26i,  0.03+0.10i,  0.05+0.17i },
                       {  0.27-0.26i,  0.59+0.00i,  0.14+0.07i,  0.24+0.14i },
                       {  0.03-0.10i,  0.14-0.07i,  0.04+0.00i,  0.08+0.00i },
                       {  0.05-0.17i,  0.24-0.14i,  0.08-0.00i,  0.13+0.00i }  };
			Ms[2] = {  {  0.11+0.00i, -0.06-0.11i,  0.08-0.15i, -0.10+0.20i },
					   { -0.06+0.11i,  0.14+0.00i,  0.11+0.16i, -0.15-0.21i },
                       {  0.08+0.15i,  0.11-0.16i,  0.27+0.00i, -0.36+0.01i },
                       { -0.10-0.20i, -0.15+0.21i, -0.36-0.01i,  0.49+0.00i } };
			Ms[3] = {  {  0.27+0.00i,  0.07-0.03i, -0.28-0.21i, -0.12-0.23i },
					   {  0.07+0.03i,  0.02+0.00i, -0.05-0.09i, -0.00-0.08i },
					   { -0.28+0.21i, -0.05+0.09i,  0.45+0.00i,  0.31+0.15i },
                       { -0.12+0.23i, -0.00+0.08i,  0.31-0.15i,  0.26+0.00i } };
			Ms[4] = {  {  0.10+0.00i, -0.18-0.07i,  0.02+0.10i, -0.21-0.02i },
                       { -0.18+0.07i,  0.37+0.00i, -0.11-0.16i,  0.38-0.11i },
                       {  0.02-0.10i, -0.11+0.16i,  0.10+0.00i, -0.07+0.20i },
                       { -0.21+0.02i,  0.38+0.11i, -0.07-0.20i,  0.43+0.00i } };
			Ms[5] = {  {  0.26+0.00i,  0.09-0.12i,  0.38-0.12i,  0.08-0.05i },
                       {  0.09+0.12i,  0.09+0.00i,  0.19+0.14i,  0.05+0.02i },
                       {  0.38+0.12i,  0.19-0.14i,  0.62+0.00i,  0.15-0.04i },
                       {  0.08+0.05i,  0.05-0.02i,  0.15+0.04i,  0.04+0.00i } };
			Ms[6] = {  {  0.06+0.00i,  0.07+0.17i, -0.01-0.02i, -0.08-0.13i },
                       {  0.07-0.17i,  0.54+0.00i, -0.06-0.00i, -0.45+0.08i },
                       { -0.01+0.02i, -0.06+0.00i,  0.01+0.00i,  0.05-0.01i },
                       { -0.08+0.13i, -0.45-0.08i,  0.05+0.01i,  0.39+0.00i } };
			Ms[7] = {  {  0.58+0.00i,  0.03+0.02i, -0.40+0.04i,  0.21+0.20i },
                       {  0.03-0.02i,  0.00+0.00i, -0.02+0.02i,  0.02+0.00i },
                       { -0.40-0.04i, -0.02-0.02i,  0.27+0.00i, -0.13-0.15i },
                       {  0.21-0.20i,  0.02-0.00i, -0.13+0.15i,  0.15+0.00i } };
			Ms[8] = {  {  0.30+0.00i, -0.39+0.02i, -0.13-0.18i, -0.03-0.05i },
                       { -0.39-0.02i,  0.52+0.00i,  0.16+0.25i,  0.04+0.07i },
                       { -0.13+0.18i,  0.16-0.25i,  0.17+0.00i,  0.05+0.00i },
                       { -0.03+0.05i,  0.04-0.07i,  0.05-0.00i,  0.01+0.00i } };
			Ms[9] = {  {  0.34+0.00i,  0.22+0.01i,  0.04+0.13i,  0.15-0.37i },
                       {  0.22-0.01i,  0.15+0.00i,  0.03+0.08i,  0.09-0.25i },
                       {  0.04-0.13i,  0.03-0.08i,  0.05+0.00i, -0.12-0.10i },
                       {  0.15+0.37i,  0.09+0.25i, -0.12+0.10i,  0.46+0.00i } };
			Ms[10] = { {  0.08+0.00i,  0.13+0.07i,  0.04-0.20i, -0.03+0.08i },
                       {  0.13-0.07i,  0.29+0.00i, -0.10-0.38i,  0.01+0.16i },
                       {  0.04+0.20i, -0.10+0.38i,  0.54+0.00i, -0.22-0.05i },
                       { -0.03-0.08i,  0.01-0.16i, -0.22+0.05i,  0.09+0.00i } };
			Ms[11] = { {  0.29+0.00i,  0.03-0.10i,  0.06+0.26i, -0.08+0.34i },
                       {  0.03+0.10i,  0.04+0.00i, -0.09+0.05i, -0.13+0.01i },
                       {  0.06-0.26i, -0.09-0.05i,  0.24+0.00i,  0.29+0.14i },
                       { -0.08-0.34i, -0.13-0.01i,  0.29-0.14i,  0.43+0.00i } };
			Ms[12] = { {  0.03+0.00i, -0.03+0.04i, -0.02-0.08i, -0.12+0.00i },
                       { -0.03-0.04i,  0.09+0.00i, -0.09+0.12i,  0.15+0.18i },
                       { -0.02+0.08i, -0.09-0.12i,  0.27+0.00i,  0.10-0.40i },
                       { -0.12-0.00i,  0.15-0.18i,  0.10+0.40i,  0.62+0.00i } };
			Ms[13] = { {  0.19+0.00i, -0.03+0.14i, -0.08+0.31i, -0.10-0.13i },
                       { -0.03-0.14i,  0.11+0.00i,  0.25+0.01i, -0.08+0.10i },
                       { -0.08-0.31i,  0.25-0.01i,  0.56+0.00i, -0.17+0.22i },
                       { -0.10+0.13i, -0.08-0.10i, -0.17-0.22i,  0.14+0.00i } };
			Ms[14] = { {  0.37+0.00i, -0.11+0.27i,  0.16-0.19i,  0.28+0.10i },
                       { -0.11-0.27i,  0.23+0.00i, -0.19-0.05i, -0.02-0.23i },
                       {  0.16+0.19i, -0.19+0.05i,  0.16+0.00i,  0.07+0.18i },
                       {  0.28-0.10i, -0.02+0.23i,  0.07-0.18i,  0.23+0.00i } };
			Ms[15] = { {  0.41+0.00i,  0.18-0.45i, -0.05-0.04i, -0.05+0.03i },
                       {  0.18+0.45i,  0.57+0.00i,  0.03-0.07i, -0.05-0.05i },
                       { -0.05+0.04i,  0.03+0.07i,  0.01+0.00i,  0.00-0.01i },
                       { -0.05-0.03i, -0.05+0.05i,  0.00+0.01i,  0.01+0.00i } };
		} else if (d == 6 && sets == 4) {
			Ms[0] = {  {  0.17+0.00i, -0.11+0.01i,  0.09+0.09i, -0.03+0.17i,  0.12-0.16i,  0.18-0.13i },
                       { -0.11-0.01i,  0.06+0.00i, -0.05-0.06i,  0.02-0.10i, -0.08+0.09i, -0.11+0.07i },
                       {  0.09-0.09i, -0.05+0.06i,  0.09+0.00i,  0.07+0.10i, -0.01-0.15i,  0.03-0.16i },
                       { -0.03-0.17i,  0.02+0.10i,  0.07-0.10i,  0.16+0.00i, -0.17-0.09i, -0.15-0.15i },
                       {  0.12+0.16i, -0.08-0.09i, -0.01+0.15i, -0.17+0.09i,  0.23+0.00i,  0.24+0.06i },
                       {  0.18+0.13i, -0.11-0.07i,  0.03+0.16i, -0.15+0.15i,  0.24-0.06i,  0.28+0.00i } };
			Ms[1] = {  {  0.07+0.00i,  0.11-0.12i, -0.05-0.07i,  0.12-0.06i,  0.07-0.02i,  0.05-0.03i },
                       {  0.11+0.12i,  0.42+0.00i,  0.05-0.23i,  0.30+0.13i,  0.16+0.09i,  0.15+0.03i },
                       { -0.05+0.07i,  0.05+0.23i,  0.13+0.00i, -0.03+0.18i, -0.03+0.09i, -0.00+0.08i },
                       {  0.12+0.06i,  0.30-0.13i, -0.03-0.18i,  0.25+0.00i,  0.14+0.02i,  0.11-0.02i },
                       {  0.07+0.02i,  0.16-0.09i, -0.03-0.09i,  0.14-0.02i,  0.08+0.00i,  0.06-0.02i },
                       {  0.05+0.03i,  0.15-0.03i, -0.00-0.08i,  0.11+0.02i,  0.06+0.02i,  0.05+0.00i } };
			Ms[2] = {  {  0.06+0.00i,  0.03+0.03i,  0.08-0.11i, -0.13-0.09i,  0.04+0.01i, -0.02-0.05i },
                       {  0.03-0.03i,  0.04+0.00i, -0.02-0.12i, -0.14+0.03i,  0.03-0.02i, -0.04-0.02i },
                       {  0.08+0.11i, -0.02+0.12i,  0.34+0.00i, -0.01-0.40i,  0.05+0.10i,  0.07-0.11i },
                       { -0.13+0.09i, -0.14-0.03i, -0.01+0.40i,  0.48+0.00i, -0.11+0.06i,  0.13+0.09i },
                       {  0.04-0.01i,  0.03+0.02i,  0.05-0.10i, -0.11-0.06i,  0.03+0.00i, -0.02-0.04i },
                       { -0.02+0.05i, -0.04+0.02i,  0.07+0.11i,  0.13-0.09i, -0.02+0.04i,  0.05+0.00i } };
			Ms[3] = {  {  0.04+0.00i,  0.05-0.04i,  0.06+0.01i, -0.02-0.02i, -0.09-0.11i,  0.02+0.12i },
                       {  0.05+0.04i,  0.08+0.00i,  0.06+0.07i, -0.01-0.04i, -0.01-0.19i, -0.08+0.15i },
                       {  0.06-0.01i,  0.06-0.07i,  0.09+0.00i, -0.04-0.02i, -0.16-0.12i,  0.06+0.16i },
                       { -0.02+0.02i, -0.01+0.04i, -0.04+0.02i,  0.02+0.00i,  0.08+0.02i, -0.05-0.05i },
                       { -0.09+0.11i, -0.01+0.19i, -0.16+0.12i,  0.08-0.02i,  0.43+0.00i, -0.32-0.20i },
                       {  0.02-0.12i, -0.08-0.15i,  0.06-0.16i, -0.05+0.05i, -0.32+0.20i,  0.33+0.00i } };
			Ms[4] = {  {  0.13+0.00i,  0.17-0.03i, -0.03+0.19i, -0.08+0.02i, -0.11+0.10i, -0.06-0.12i },
                       {  0.17+0.03i,  0.23+0.00i, -0.08+0.24i, -0.10+0.01i, -0.16+0.11i, -0.05-0.18i },
                       { -0.03-0.19i, -0.08-0.24i,  0.28+0.00i,  0.05+0.11i,  0.17+0.14i, -0.17+0.12i },
                       { -0.08-0.02i, -0.10-0.01i,  0.05-0.11i,  0.05+0.00i,  0.08-0.04i,  0.02+0.08i },
                       { -0.11-0.10i, -0.16-0.11i,  0.17-0.14i,  0.08+0.04i,  0.17+0.00i, -0.05+0.15i },
                       { -0.06+0.12i, -0.05+0.18i, -0.17-0.12i,  0.02-0.08i, -0.05-0.15i,  0.15+0.00i } };
			Ms[5] = {  {  0.53+0.00i, -0.26+0.15i, -0.15-0.10i,  0.14-0.02i, -0.04+0.18i, -0.16+0.22i },
                       { -0.26-0.15i,  0.16+0.00i,  0.05+0.09i, -0.08-0.03i,  0.07-0.07i,  0.14-0.06i },
                       { -0.15+0.10i,  0.05-0.09i,  0.06+0.00i, -0.04+0.03i, -0.02-0.06i,  0.01-0.09i },
                       {  0.14+0.02i, -0.08+0.03i, -0.04-0.03i,  0.04+0.00i, -0.02+0.05i, -0.05+0.05i },
                       { -0.04-0.18i,  0.07+0.07i, -0.02+0.06i, -0.02-0.05i,  0.06+0.00i,  0.09+0.04i },
                       { -0.16-0.22i,  0.14+0.06i,  0.01+0.09i, -0.05-0.05i,  0.09-0.04i,  0.14+0.00i } };
			Ms[6] = {  {  0.21+0.00i, -0.14-0.13i,  0.21+0.18i,  0.07-0.10i, -0.04-0.13i, -0.14-0.04i },
                       { -0.14+0.13i,  0.17+0.00i, -0.25+0.01i,  0.02+0.11i,  0.10+0.06i,  0.12-0.07i },
                       {  0.21-0.18i, -0.25-0.01i,  0.36+0.00i, -0.02-0.15i, -0.15-0.10i, -0.18+0.09i },
                       {  0.07+0.10i,  0.02-0.11i, -0.02+0.15i,  0.07+0.00i,  0.05-0.06i, -0.03-0.08i },
                       { -0.04+0.13i,  0.10-0.06i, -0.15+0.10i,  0.05+0.06i,  0.08+0.00i,  0.05-0.08i },
                       { -0.14+0.04i,  0.12+0.07i, -0.18-0.09i, -0.03+0.08i,  0.05+0.08i,  0.11+0.00i } };
			Ms[7] = {  {  0.18+0.00i,  0.22+0.04i, -0.07-0.08i, -0.01-0.02i,  0.02+0.08i, -0.27+0.05i },
                       {  0.22-0.04i,  0.28+0.00i, -0.11-0.09i, -0.02-0.02i,  0.04+0.09i, -0.33+0.12i },
                       { -0.07+0.08i, -0.11+0.09i,  0.07+0.00i,  0.01+0.00i, -0.05-0.02i,  0.09-0.15i },
                       { -0.01+0.02i, -0.02+0.02i,  0.01-0.00i,  0.00+0.00i, -0.01-0.00i,  0.01-0.03i },
                       {  0.02-0.08i,  0.04-0.09i, -0.05+0.02i, -0.01+0.00i,  0.04+0.00i, -0.01+0.13i },
                       { -0.27-0.05i, -0.33-0.12i,  0.09+0.15i,  0.01+0.03i, -0.01-0.13i,  0.42+0.00i } };
			Ms[8] = {  {  0.43+0.00i, -0.12-0.11i, -0.22-0.22i, -0.16-0.03i, -0.05-0.04i,  0.26-0.13i },
                       { -0.12+0.11i,  0.07+0.00i,  0.12+0.00i,  0.05-0.03i,  0.02-0.00i, -0.04+0.11i },
                       { -0.22+0.22i,  0.12-0.00i,  0.23+0.00i,  0.10-0.07i,  0.04-0.01i, -0.07+0.21i },
                       { -0.16+0.03i,  0.05+0.03i,  0.10+0.07i,  0.06+0.00i,  0.02+0.01i, -0.09+0.07i },
                       { -0.05+0.04i,  0.02+0.00i,  0.04+0.01i,  0.02-0.01i,  0.01+0.00i, -0.02+0.04i },
                       {  0.26+0.13i, -0.04-0.11i, -0.07-0.21i, -0.09-0.07i, -0.02-0.04i,  0.20+0.00i } };
			Ms[9] = {  {  0.09+0.00i,  0.01+0.16i, -0.02+0.03i,  0.12+0.09i,  0.01-0.13i,  0.06+0.10i },
                       {  0.01-0.16i,  0.30+0.00i,  0.05+0.04i,  0.19-0.21i, -0.24-0.04i,  0.19-0.09i },
                       { -0.02-0.03i,  0.05-0.04i,  0.01+0.00i,  0.00-0.06i, -0.04+0.02i,  0.02-0.04i },
                       {  0.12-0.09i,  0.19+0.21i,  0.00+0.06i,  0.26+0.00i, -0.12-0.19i,  0.18+0.07i },
                       {  0.01+0.13i, -0.24+0.04i, -0.04-0.02i, -0.12+0.19i,  0.19+0.00i, -0.14+0.10i },
                       {  0.06-0.10i,  0.19+0.09i,  0.02+0.04i,  0.18-0.07i, -0.14-0.10i,  0.15+0.00i } };
			Ms[10] = { {  0.09+0.00i, -0.01+0.03i,  0.08+0.07i,  0.04+0.03i,  0.05+0.23i,  0.08+0.02i },
                       { -0.01-0.03i,  0.01+0.00i,  0.02-0.04i,  0.01-0.02i,  0.09-0.04i,  0.00-0.04i },
                       {  0.08-0.07i,  0.02+0.04i,  0.14+0.00i,  0.07-0.01i,  0.24+0.17i,  0.10-0.05i },
                       {  0.04-0.03i,  0.01+0.02i,  0.07+0.01i,  0.04+0.00i,  0.11+0.10i,  0.05-0.02i },
                       {  0.05-0.23i,  0.09+0.04i,  0.24-0.17i,  0.11-0.10i,  0.64+0.00i,  0.11-0.21i },
                       {  0.08-0.02i,  0.00+0.04i,  0.10+0.05i,  0.05+0.02i,  0.11+0.21i,  0.09+0.00i } };
			Ms[11] = { {  0.01+0.00i,  0.03+0.01i,  0.03+0.02i, -0.06+0.02i, -0.00-0.02i,  0.01-0.00i },
                       {  0.03-0.01i,  0.17+0.00i,  0.16+0.07i, -0.25+0.18i, -0.02-0.08i,  0.06-0.03i },
                       {  0.03-0.02i,  0.16-0.07i,  0.19+0.00i, -0.17+0.28i, -0.05-0.06i,  0.04-0.06i },
                       { -0.06-0.02i, -0.25-0.18i, -0.17-0.28i,  0.57+0.00i, -0.05+0.14i, -0.12-0.01i },
                       { -0.00+0.02i, -0.02+0.08i, -0.05+0.06i, -0.05-0.14i,  0.04+0.00i,  0.01+0.03i },
                       {  0.01+0.00i,  0.06+0.03i,  0.04+0.06i, -0.12+0.01i,  0.01-0.03i,  0.03+0.00i } };
			Ms[12] = { {  0.07+0.00i, -0.20-0.12i,  0.01+0.04i,  0.05-0.00i,  0.05+0.08i,  0.05-0.02i },
                       { -0.20+0.12i,  0.72+0.00i, -0.09-0.09i, -0.13+0.09i, -0.25-0.15i, -0.09+0.13i },
                       {  0.01-0.04i, -0.09+0.09i,  0.02+0.00i,  0.00-0.03i,  0.05-0.01i, -0.00-0.03i },
                       {  0.05+0.00i, -0.13-0.09i,  0.00+0.03i,  0.03+0.00i,  0.03+0.06i,  0.03-0.01i },
                       {  0.05-0.08i, -0.25+0.15i,  0.05+0.01i,  0.03-0.06i,  0.12+0.00i,  0.01-0.06i },
                       {  0.05+0.02i, -0.09-0.13i, -0.00+0.03i,  0.03+0.01i,  0.01+0.06i,  0.03+0.00i } };
			Ms[13] = { {  0.12+0.00i, -0.03+0.02i, -0.05-0.07i, -0.07-0.21i, -0.21+0.05i, -0.05-0.02i },
                       { -0.03-0.02i,  0.01+0.00i,  0.00+0.03i, -0.02+0.07i,  0.07+0.02i,  0.01+0.01i },
                       { -0.05+0.07i,  0.00-0.03i,  0.06+0.00i,  0.15+0.04i,  0.06-0.14i,  0.03-0.02i },
                       { -0.07+0.21i, -0.02-0.07i,  0.15-0.04i,  0.41+0.00i,  0.04-0.39i,  0.06-0.07i },
                       { -0.21-0.05i,  0.07-0.02i,  0.06+0.14i,  0.04+0.39i,  0.38+0.00i,  0.07+0.05i },
                       { -0.05+0.02i,  0.01-0.01i,  0.03+0.02i,  0.06+0.07i,  0.07-0.05i,  0.02+0.00i } };
			Ms[14] = { {  0.10+0.00i, -0.01-0.01i, -0.22+0.08i, -0.04-0.02i,  0.06-0.15i,  0.03+0.07i },
                       { -0.01+0.01i,  0.00+0.00i,  0.02-0.02i,  0.00-0.00i,  0.00+0.02i, -0.01-0.01i },
                       { -0.22-0.08i,  0.02+0.02i,  0.56+0.00i,  0.07+0.07i, -0.27+0.28i,  0.00-0.18i },
                       { -0.04+0.02i,  0.00+0.00i,  0.07-0.07i,  0.02+0.00i,  0.00+0.07i, -0.02-0.02i },
                       {  0.06+0.15i,  0.00-0.02i, -0.27-0.28i,  0.00-0.07i,  0.27+0.00i, -0.09+0.09i },
                       {  0.03-0.07i, -0.01+0.01i,  0.00+0.18i, -0.02+0.02i, -0.09-0.09i,  0.06+0.00i } };
			Ms[15] = { {  0.02+0.00i,  0.01-0.03i, -0.00+0.00i, -0.04+0.01i,  0.02-0.03i, -0.04-0.11i },
                       {  0.01+0.03i,  0.07+0.00i, -0.01-0.00i, -0.05-0.07i,  0.06+0.02i,  0.18-0.14i },
                       { -0.00-0.00i, -0.01+0.00i,  0.00+0.00i,  0.01+0.00i, -0.01+0.00i, -0.01+0.02i },
                       { -0.04-0.01i, -0.05+0.07i,  0.01-0.00i,  0.10+0.00i, -0.06+0.04i,  0.00+0.28i },
                       {  0.02+0.03i,  0.06-0.02i, -0.01-0.00i, -0.06-0.04i,  0.06+0.00i,  0.12-0.17i },
                       { -0.04+0.11i,  0.18+0.14i, -0.01-0.02i,  0.00-0.28i,  0.12+0.17i,  0.76+0.00i } };
			Ms[16] = { {  0.08+0.00i,  0.03+0.10i, -0.10-0.00i,  0.14+0.11i, -0.06+0.09i,  0.01-0.08i },
                       {  0.03-0.10i,  0.13+0.00i, -0.04+0.13i,  0.20-0.13i,  0.08+0.11i, -0.10-0.05i },
                       { -0.10+0.00i, -0.04-0.13i,  0.14+0.00i, -0.19-0.15i,  0.08-0.12i, -0.01+0.11i },
                       {  0.14-0.11i,  0.20+0.13i, -0.19+0.15i,  0.42+0.00i,  0.01+0.25i, -0.10-0.16i },
                       { -0.06-0.09i,  0.08-0.11i,  0.08+0.12i,  0.01-0.25i,  0.14+0.00i, -0.10+0.05i },
                       {  0.01+0.08i, -0.10+0.05i, -0.01-0.11i, -0.10+0.16i, -0.10-0.05i,  0.09+0.00i } };
			Ms[17] = { {  0.61+0.00i,  0.20+0.04i,  0.36-0.05i, -0.05+0.10i,  0.15-0.04i,  0.00+0.16i },
                       {  0.20-0.04i,  0.07+0.00i,  0.12-0.04i, -0.01+0.04i,  0.05-0.02i,  0.01+0.05i },
                       {  0.36+0.05i,  0.12+0.04i,  0.22+0.00i, -0.04+0.06i,  0.09-0.01i, -0.01+0.09i },
                       { -0.05-0.10i, -0.01-0.04i, -0.04-0.06i,  0.02+0.00i, -0.02-0.02i,  0.03-0.01i },
                       {  0.15+0.04i,  0.05+0.02i,  0.09+0.01i, -0.02+0.02i,  0.04+0.00i, -0.01+0.04i },
                       {  0.00-0.16i,  0.01-0.05i, -0.01-0.09i,  0.03+0.01i, -0.01-0.04i,  0.04+0.00i } };
			Ms[18] = { {  0.27+0.00i,  0.01-0.14i, -0.11-0.03i,  0.06+0.26i,  0.25+0.11i, -0.09-0.10i },
                       {  0.01+0.14i,  0.08+0.00i,  0.01-0.06i, -0.13+0.04i, -0.05+0.14i,  0.05-0.05i },
                       { -0.11+0.03i,  0.01+0.06i,  0.05+0.00i, -0.06-0.10i, -0.12-0.02i,  0.05+0.03i },
                       {  0.06-0.26i, -0.13-0.04i, -0.06+0.10i,  0.26+0.00i,  0.16-0.22i, -0.11+0.07i },
                       {  0.25-0.11i, -0.05-0.14i, -0.12+0.02i,  0.16+0.22i,  0.28+0.00i, -0.12-0.05i },
                       { -0.09+0.10i,  0.05+0.05i,  0.05-0.03i, -0.11-0.07i, -0.12+0.05i,  0.06+0.00i } };
			Ms[19] = { {  0.00+0.00i, -0.01+0.01i,  0.01+0.01i,  0.00-0.01i,  0.00+0.02i, -0.02+0.00i },
                       { -0.01-0.01i,  0.14+0.00i, -0.03-0.10i, -0.08+0.07i,  0.12-0.18i,  0.19+0.13i },
                       {  0.01-0.01i, -0.03+0.10i,  0.07+0.00i, -0.04-0.07i,  0.10+0.11i, -0.13+0.10i },
                       {  0.00+0.01i, -0.08-0.07i, -0.04+0.07i,  0.08+0.00i, -0.16+0.03i, -0.03-0.17i },
                       {  0.00-0.02i,  0.12+0.18i,  0.10-0.11i, -0.16-0.03i,  0.33+0.00i, -0.01+0.35i },
                       { -0.02-0.00i,  0.19-0.13i, -0.13-0.10i, -0.03+0.17i, -0.01-0.35i,  0.38+0.00i } };
			Ms[20] = { {  0.29+0.00i, -0.06+0.22i, -0.02+0.01i, -0.21+0.06i, -0.17-0.23i, -0.15-0.02i },
                       { -0.06-0.22i,  0.17+0.00i,  0.01+0.01i,  0.09+0.15i, -0.14+0.17i,  0.01+0.12i },
                       { -0.02-0.01i,  0.01-0.01i,  0.00+0.00i,  0.02+0.00i,  0.00+0.02i,  0.01+0.01i },
                       { -0.21-0.06i,  0.09-0.15i,  0.02-0.00i,  0.17+0.00i,  0.07+0.20i,  0.10+0.05i },
                       { -0.17+0.23i, -0.14-0.17i,  0.00-0.02i,  0.07-0.20i,  0.28+0.00i,  0.10-0.11i },
                       { -0.15+0.02i,  0.01-0.12i,  0.01-0.01i,  0.10-0.05i,  0.10+0.11i,  0.08+0.00i } };
			Ms[21] = { {  0.32+0.00i,  0.10-0.15i, -0.02+0.22i, -0.00-0.28i, -0.05+0.08i,  0.17+0.14i },
                       {  0.10+0.15i,  0.10+0.00i, -0.11+0.06i,  0.13-0.09i, -0.05+0.00i, -0.01+0.12i },
                       { -0.02-0.22i, -0.11-0.06i,  0.15+0.00i, -0.19+0.02i,  0.06+0.03i,  0.09-0.13i },
                       { -0.00+0.28i,  0.13+0.09i, -0.19-0.02i,  0.24+0.00i, -0.07-0.04i, -0.12+0.15i },
                       { -0.05-0.08i, -0.05-0.00i,  0.06-0.03i, -0.07+0.04i,  0.03+0.00i,  0.01-0.07i },
                       {  0.17-0.14i, -0.01-0.12i,  0.09+0.13i, -0.12-0.15i,  0.01+0.07i,  0.16+0.00i } };
			Ms[22] = { {  0.11+0.00i, -0.07+0.11i,  0.11-0.17i,  0.15-0.02i, -0.03+0.04i,  0.12-0.02i },
                       { -0.07-0.11i,  0.15+0.00i, -0.24+0.01i, -0.12-0.13i,  0.06-0.00i, -0.11-0.11i },
                       {  0.11+0.17i, -0.24-0.01i,  0.36+0.00i,  0.17+0.22i, -0.08-0.00i,  0.15+0.17i },
                       {  0.15+0.02i, -0.12+0.13i,  0.17-0.22i,  0.21+0.00i, -0.04+0.05i,  0.17-0.01i },
                       { -0.03-0.04i,  0.06+0.00i, -0.08+0.00i, -0.04-0.05i,  0.02+0.00i, -0.04-0.04i },
                       {  0.12+0.02i, -0.11+0.11i,  0.15-0.17i,  0.17+0.01i, -0.04+0.04i,  0.15+0.00i } };
			Ms[23] = { {  0.01+0.00i,  0.03-0.04i,  0.04-0.03i,  0.00-0.02i, -0.01-0.02i, -0.04-0.00i },
                       {  0.03+0.04i,  0.35+0.00i,  0.35+0.08i,  0.11-0.04i,  0.07-0.13i, -0.13-0.22i },
                       {  0.04+0.03i,  0.35-0.08i,  0.36+0.00i,  0.10-0.07i,  0.04-0.14i, -0.17-0.19i },
                       {  0.00+0.02i,  0.11+0.04i,  0.10+0.07i,  0.04+0.00i,  0.04-0.03i, -0.01-0.08i },
                       { -0.01+0.02i,  0.07+0.13i,  0.04+0.14i,  0.04+0.03i,  0.06+0.00i,  0.06-0.09i },
                       { -0.04+0.00i, -0.13+0.22i, -0.17+0.19i, -0.01+0.08i,  0.06+0.09i,  0.18+0.00i } };

		} else {
			std::cerr << "Don't have a nearby point for this problem" << std::endl;
			return 0;

		}

		// Put these in the B diagonals
		for (int i=0; i<numBMats; i++) {
			int matLoc = (numRhoMats+i)*2*d;
			for (int j=0; j<d; j++) {
				for (int k=0; k<d; k++) {
					XDense(matLoc+j, matLoc+k) = std::real(Ms[i][j][k]);
					XDense(matLoc+d+j, matLoc+d+k) = std::real(Ms[i][j][k]);
					XDense(matLoc+j, matLoc+d+k) = std::imag(Ms[i][j][k]);
					XDense(matLoc+d+j, matLoc+k) = -std::imag(Ms[i][j][k]);
				}
			}
		}

		// Calculate the ideal rho TODO
		double rtd = std::sqrt(d);
		int rhoLoc = 0;
		for (int i=0; i<numMeasureB; i++) {
			for (int j=i+1; j<numMeasureB; j++) {
				for (int k=0; k<numOutcomeB; k++) {
					for (int l=0; l<numOutcomeB; l++) {
						int loc1 = (numRhoMats + i*numOutcomeB + k)*d*2;
						int loc2 = (numRhoMats + j*numOutcomeB + l)*d*2;
						Eigen::MatrixXd B1 = XDense.block(loc1, loc1, d, d);
						Eigen::MatrixXd B2 = XDense.block(loc2, loc2, d, d);
						Eigen::MatrixXd newRho = B1 + B2 + rtd*(B1*B2 + B2*B1);
						XDense.block(rhoLoc, rhoLoc, d, d) = newRho / newRho.trace();
						rhoLoc += d;
					}
				}
			}
		}

		// Then this X into an x
		x = matToVec(XDense.sparseView());

	} 

	//x(0) *= 1.1;
	//x(7) *= 1.1;
	//x(9) *= 1.1;
	
	// TODO
	std::cout << "initial f(x) = " << f(x) << " <= " << maxVal << std::endl;
	std::cout << "initial g(x) = " << g(x) << std::endl;

	return 0;

	// Gradient descent to make sure we start with an interior point
	Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
	for (int i=0; i<10000; i++) {
		v = g(x);
		if (outputMode == "") {
			std::cout << "|g(x)| = " << v.norm() << std::endl;
		}
		if (v.norm() < gThresh) {
			break;
		}
		x -= 0.01*v(0)*delg(x).row(0);
		for (int j=1; j<m; j++) {
			x -= 0.1*v(j)*delg(x).row(j);
		}
	}

	// Get the full matrices from this
	X = vecToMat(x);
	XDense = Eigen::MatrixXd(X);

	std::cout << "after f(x) = " << f(x) << " <= " << maxVal << std::endl;
	std::cout << "after g(x) = " << g(x) << std::endl;

	std::cout << std::endl;
	prettyPrint("x = ", x);
	std::cout << std::endl;
	prettyPrint("X = ", X);
	std::cout << std::endl;

	// Output the initial X
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "        Initial Matrices        " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		Eigen::MatrixXd M(d, d);
		for (int i=0; i<numMats; i++) {
			int ind = i*d;
			M = Eigen::MatrixXd(XDense.block(ind, ind, d, d));
			std::cout << std::endl;
			prettyPrint("mat " + std::to_string(i) + " = ", M);
			std::cout << std::endl;
			std::cout << "|M^2-M|  = " << (M.adjoint()*M - M).squaredNorm() << std::endl;
			std::cout << "tr(M^2-M)  = " << (M.adjoint()*M - M).trace() << std::endl;
			std::cout << "is M PD? = " << isPD(M) << std::endl;
		}
	}
	std::cout << "" << std::endl;

	// Ensure this is an interior point
	//if (std::abs(g(x).norm()) > gThresh) {
		//std::cerr << "Error - X should start as an interior point (g(x) = " << g(x).norm() << " > gThresh)" << std::endl;
		//return 1;
	//}
	//if (!isPD(X)) {
		//std::cerr << "Error - X should start as an interior point (X is not semidefinite)" << std::endl;
		//return 1;
	//}

	// Initialise Z
	ZDense = Eigen::MatrixXd::Zero(p, p);
	//for (int i=0; i<numMeasureB; i++) {
		//for (int j=0; j<numOutcomeB; j++) {
			//int currentLoc = (i*numOutcomeB + j) * d;
			//int copyLoc = (i*numOutcomeB + ((j+1) % numOutcomeB)) * d;
			//ZDense.block(currentLoc,currentLoc,d,d) = XDense.block(copyLoc,copyLoc,d,d);
			//ZDense.block(currentLoc+halfP,currentLoc+halfP,d,d) = XDense.block(copyLoc,copyLoc,d,d);
			//ZDense.block(currentLoc+halfP,currentLoc,d,d) = XDense.block(copyLoc+halfP,copyLoc,d,d);
			//ZDense.block(currentLoc,currentLoc+halfP,d,d) = -X.block(copyLoc+halfP,copyLoc,d,d);
		//}
	//}

	// Output the initial Z
	//if (outputMode == "") {
		//std::cout << "" << std::endl;
		//std::cout << "--------------------------------" << std::endl;
		//std::cout << "      Initial Z " << std::endl;;
		//std::cout << "--------------------------------" << std::endl;
		//for (int i=0; i<numMeasureB; i++) {
			//for (int j=0; j<numOutcomeB; j++) {
				//int ind = (i*numOutcomeB + j)*d;
				//Eigen::MatrixXcd M = Z.block(ind, ind, d, d) + 1i*Z.block(ind+halfP, ind, d, d);
				//std::cout << std::endl;
				//prettyPrint("Z_" + std::to_string(j) + "^" + std::to_string(i) + " = ", M);
			//}
		//}
		//std::cout << std::endl;
		//prettyPrint("X dot Z = ", X.cwiseProduct(Z).sum());
	//}

	// Init some thing that are used for the first calcs
	Eigen::SparseMatrix<double> XInverse = XDense.inverse().sparseView();
	Eigen::SparseMatrix<double> ZInverse = ZDense.inverse().sparseView();
	Eigen::VectorXd gCached = g(x);
	Eigen::MatrixXd A_0 = delg(x);
	Eigen::VectorXd delfCached = delf(x);
	Eigen::VectorXd delLCached = delL(y, Z, delfCached, A_0);
	double rMagZero = rMag(0, Z, X, delLCached, gCached);

	// Used for the BFGS update
	Eigen::VectorXd prevx = x;
	Eigen::MatrixXd prevA_0 = A_0;
	Eigen::VectorXd prevDelfCached = delfCached;

	// To prevent reinit each time
	Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd AStarXInverse = Eigen::VectorXd::Zero(n);
	Eigen::MatrixXd GHInverse = Eigen::MatrixXd::Zero(n, n);
	Eigen::MatrixXd deltaX = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd deltax = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd deltay = Eigen::VectorXd::Zero(m);
	Eigen::MatrixXd deltaZ = Eigen::MatrixXd::Zero(p, p);

	// If using the BFGS update
	if (useBFGS) {

		// Only calculate the full Hessian once, use a BFGS-like update later
		G = del2L(x, y);

		// Ensure it's positive definite
		makePD(G);

	}

	// Outer loop
	double rMagMu = 0;
	int k = 0;
	int totalInner = 0;
	for (k=0; k<maxOuterIter; k++) {

		// Check if global convergence is reached
		rMagZero = rMag(0, Z, X, delLCached, gCached);
		if (rMagZero <= epsilon) {
			break;
		}

		// If anything is NaN, stop
		if (isnan(rMagZero)) {
			break;
		}

		// Outer-iteration output
		if (outputMode == "") {
			std::cout << std::endl;
			std::cout << "----------------------------------" << std::endl;
			std::cout << "         Iteration " << k << std::endl;;
			std::cout << "----------------------------------" << std::endl;
		}

		// Otherwise find the optimum for the current mu
		double epsilonPrime = M_c * mu;
		for (int k2=0; k2<maxInnerIter; k2++) {

			// Also make sure the total inner iteration count is valid
			totalInner++;
			if (totalInner > maxTotalInner) {
				break;
			}
		
			// If anything is NaN, stop
			if (isnan(rMagMu)) {
				break;
			}

			// Update the caches
			X = vecToMat(x);
			XDense = Eigen::MatrixXd(X);
			ZDense = Eigen::MatrixXd(Z);
			XInverse = XDense.inverse().sparseView();
			ZInverse = ZDense.inverse().sparseView();
			gCached = g(x);
			A_0 = delg(x);
			delfCached = delf(x);
			delLCached = delL(y, Z, delfCached, A_0);

			// If not doing BFGS, need to do a full re-calc of G
			if (!useBFGS || gCached.norm() > BFGSmaxG) {

				// Update G
				G = del2L(x, y);

				// Ensure it's positive definite
				makePD(G);

			}

			// Construct H TODO
			//H = 0.0;

			// Calculate/cache a few useful matrices 
			GHInverse = (G + H).inverse();
			//for (int i=0; i<n; i++) {
				//AStarXInverse(i) = As[i].cwiseProduct(XInverse).sum();
			//}
			AStarXInverse = matToVec(XInverse);

			// Calculate the x and y by solving system of linear equations
			Eigen::MatrixXd leftMat = Eigen::MatrixXd::Zero(n+m, n+m);
			Eigen::VectorXd rightVec = Eigen::VectorXd::Zero(n+m);
			Eigen::VectorXd solution = Eigen::VectorXd::Zero(n+m);
			leftMat.block(0,0,n,n) = G + H;
			leftMat.block(n,0,m,n) = -A_0;
			leftMat.block(0,n,n,m) = -A_0.transpose();
			rightVec.head(n) = -delfCached + A_0.transpose()*y + mu*AStarXInverse;
			rightVec.tail(m) = gCached;
			solution = leftMat.colPivHouseholderQr().solve(rightVec);
			deltax = solution.head(n);
			deltay = solution.tail(m);

			// Then calculate the Z
			deltaX = vecToMat(deltax);
			deltaZ = mu*XInverse - Z - 0.5*(XInverse*deltaX*Z + Z*deltaX*XInverse);

			// Determine the max l such that beta^l = 1e-9
			int maxL = std::log(epsilon) / std::log(beta);

			// Get a base step size using the min eigenvalues
			double alphaBarX = -gammaVal / (XInverse * deltaX).eigenvalues().real().minCoeff();
			double alphaBarZ = -gammaVal / (ZInverse * deltaZ).eigenvalues().real().minCoeff();
			if (alphaBarX < 0) {
				alphaBarX = 1;
			}
			if (alphaBarZ < 0) {
				alphaBarZ = 1;
			}
			double alphaBar = std::min(std::min(alphaBarX, alphaBarZ), 1.0);

			// Calculate optimal step size using a line search
			double alpha;
			int l;
			double FCached = F(x, Z, mu);
			double deltaFCached = deltaF(deltaZ, ZInverse, Z, X, XInverse, delfCached, gCached, A_0, deltax);
			for (l=0; l<maxL; l++){
				alpha = alphaBar * std::pow(beta, l);
				if (F(x+alpha*deltax, Z+alpha*deltaZ, mu) <= FCached + epsilonZero*alpha*deltaFCached && isPD(vecToMat(x+alpha*deltax))) {
					break;
				}
			}

			// Inner-iteration output
			rMagMu = rMag(mu, Z, X, delLCached, gCached);
			if (outputMode == "") {

				// Output the line
				std::cout << "f = " << f(x) << "   r = " << rMagMu  << " ?< " << epsilonPrime << "   g = " << gCached.norm() << std::endl;

			}

			// Check if local convergence is reached
			if (rMagMu <= epsilonPrime) {
				break;
			}

			// Save certain quantities for the BFGS update
			if (useBFGS) {
				prevx = x;
				prevDelfCached = delfCached;
				prevA_0 = A_0;
			}

			// Update variables
			x += alpha*deltax;
			y += deltay;
			Z += alpha*deltaZ.sparseView();
			
			// If using a BFGS update
			if (useBFGS) {

				// Update certain quantities
				A_0 = delg(x);
				delfCached = delf(x);

				// Update G
				Eigen::VectorXd s = x - prevx;
				Eigen::VectorXd q = delL(y, Z, delfCached, A_0) - delL(y, Z, prevDelfCached, prevA_0);
				double psi = 1;
				if ((s.adjoint()*q).real()(0) < (0.2*s.adjoint()*G*s).real()(0)) {
					psi = ((0.8*s.adjoint()*G*s) / (s.adjoint()*(G*s - q))).real()(0);
				}
				Eigen::VectorXd qBar = psi*q + (1-psi)*(G*s);
				G = G - ((G*(s*(s.transpose()*G))) / (s.adjoint()*G*s)) + ((qBar*qBar.transpose()) / (s.adjoint()*qBar));

			}
			
		}

		// Stop if the max total inner has been reached
		if (totalInner > maxTotalInner) {
			break;
		}

		// Update mu
		mu = mu / muScaling;

	}

	// Stop the timer
	auto t2 = std::chrono::high_resolution_clock::now();

	// Output the initial Z
	//if (outputMode == "") {
		//std::cout << "" << std::endl;
		//std::cout << "--------------------------------" << std::endl;
		//std::cout << "      Final Z " << std::endl;;
		//std::cout << "--------------------------------" << std::endl;
		//for (int i=0; i<numMeasureB; i++) {
			//for (int j=0; j<numOutcomeB; j++) {
				//int ind = (i*numOutcomeB + j)*d;
				//Eigen::MatrixXcd M = Z.block(ind, ind, d, d) + 1i*Z.block(ind+halfP, ind, d, d);
				//std::cout << std::endl;
				//prettyPrint("Z_" + std::to_string(j) + "^" + std::to_string(i) + " = ", M);
			//}
		//}
	//}

	// Extract the solution from X
	//if (outputMode == "") {
		//std::cout << "" << std::endl;
		//std::cout << "----------------------------------" << std::endl;
		//std::cout << "         Final Matrices " << std::endl;;
		//std::cout << "----------------------------------" << std::endl;
		//Eigen::MatrixXcd M(d, d);
		//for (int i=0; i<numMeasureB; i++) {
			//for (int j=0; j<numOutcomeB; j++) {
				//int ind = (i*numOutcomeB + j)*d;
				//M = Eigen::MatrixXcd(X.block(ind, ind, d, d));
				//std::cout << std::endl;
				//prettyPrint("B_" + std::to_string(j) + "^" + std::to_string(i) + " = ", M);
				//std::cout << std::endl;
				//std::cout << "|B^2-B|  = " << (M.adjoint()*M - M).squaredNorm() << std::endl;
				//std::cout << "is B PD? = " << isPD(M) << std::endl;
			//}
		//}
	//}

	// Final output
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "         Final Output " << std::endl;;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "        |r(w)| = " << rMagZero << " < " << epsilon << std::endl;;
		std::cout << "         -f(x) = " << -f(x) << " <= " << maxVal << std::endl;;
		std::cout << "        |g(x)| = " << gCached.norm() << std::endl;;
		std::cout << "         -L(w) = " << -L(x, X, y, Z) << std::endl;;
		std::cout << "         <X,Z> = " << X.cwiseProduct(Z).sum() << std::endl;;
		std::cout << "           |y| = " << y.norm() << std::endl;;
		std::cout << "      y^T*g(x) = " << y.transpose()*g(x) << std::endl;;
		std::cout << "     |delf(x)| = " << delfCached.norm() << std::endl;;
		std::cout << "     |delL(w)| = " << delLCached.norm() << std::endl;;
		std::cout << "     |delg(x)| = " << A_0.norm() << std::endl;;
		std::cout << "    |del2f(x)| = " << del2f(x).norm() << std::endl;;
		std::cout << "    |del2L(w)| = " << G.norm() << std::endl;;
		std::cout << "    |del2g(x)| = " << norm3D(del2g(x)) << std::endl;;
		std::cout << "   total inner = " << totalInner << std::endl;;
		std::cout << "    time taken = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;

	// Benchmarking mode
	} else if (outputMode == "B") {
		std::cout << totalInner << std::endl;

	}

	// Everything went fine
	return 0;

}

