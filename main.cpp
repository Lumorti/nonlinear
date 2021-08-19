#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <chrono>
#include <iomanip>
#include <math.h>

// Allow use of "2i" for complex
using namespace std::complex_literals;

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
double extraDiag = 10.0;
int maxOuterIter =  100000000;
int maxInnerIter =  100000000;
int maxTotalInner = 100000000;
double muScaling = 10;
double fScaling = 1.00;
double gScaling = 0.00001;
double gThresh = 1e-8;

// Parameters between 0 and 1
double gammaVal = 0.9;
double epsilonZero = 0.9;
double beta = 0.1;

// Parameters greater than 0
double epsilon = 1e-5; 
double M_c = 10000000;
double mu = 1.0;
double nu = 0.9;
double rho = 0.5;

// Useful quantities
int numPerm = 0;
int numMeasureB = 0;
int numOutcomeB = 0;
int numUniquePer = 0;
int numRealPer = 0;
int numImagPer = 0;
int numTotalPer = 0;
std::vector<Eigen::SparseMatrix<double>> As;
Eigen::SparseMatrix<double> B;

// Sizes of matrices
int n = 0;
int m = 0;
int p = 0;
int halfP = p / 2;

// For printing
int precision = 5;
std::string outputMode = "";

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

// Efficiently calculate the trace of a sparse matrix
double trace(Eigen::SparseMatrix<double> A) {

	// Start with zero
	double total = 0;

	// Loop over all non-zero elements
	for (int k=0; k<A.outerSize(); ++k) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it) {

			// If it's on the diagonal, add it
			if (it.row() == it.col()) {
				total += it.value();
			}

		}
	}

	// Return the sum of the diagonal
	return total;

}

// Function turning X to x
Eigen::VectorXd Xtox(Eigen::MatrixXd XCached) {

	// The empty vector to fill
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

	// For each submatrix in the X
	int ind = 0;
	int nextX = 0;
	int nextY = 0;
	for (int i=0; i<numMeasureB; i++) {
		for (int j=0; j<numOutcomeB-1; j++) {

			int matLoc = (i*numOutcomeB + j) * d;

			// For each real element of this matrix in the vector
			nextX = 0;
			nextY = 0;
			for (int k=0; k<numRealPer; k++) {

				// Extract it
				x(ind) = XCached(matLoc+nextX, matLoc+nextY);

				// Update the location in this submatrix
				nextX += 1;
				if (nextX >= d) {
					nextY += 1;
					nextX = nextY;
				}

				// Next element in the vector
				ind += 1;

			}

			// For each imag element of this matrix in the vector
			nextX = 1;
			nextY = 0;
			for (int k=0; k<numImagPer; k++) {

				// Extract it
				x(ind) = XCached(matLoc+nextX, matLoc+halfP+nextY);

				// Update the location in this submatrix
				nextX += 1;
				if (nextX >= d) {
					nextY += 1;
					nextX = nextY+1;
				}

				// Next element in the vector
				ind += 1;

			}

		}
	}

	// Return this new vector
	return x;

}

// Function turning x to X
Eigen::SparseMatrix<double> X(Eigen::VectorXd x, double extra=extraDiag) {

	// Create a blank p by p matrix
	Eigen::SparseMatrix<double> newX(p, p);

	// For each vector element, multiply by the corresponding A
	for (int i=0; i<n; i++) {
		newX += As[i] * x(i);
	}

	// Add the B
	newX += B;

	// Add a bit extra to make it reversible
	newX += Eigen::MatrixXd::Identity(p, p).sparseView() * extra;
	
	// Return this new matrix
	return newX;

}

// Function turning x to X without any B addition
Eigen::SparseMatrix<double> XNoB(Eigen::VectorXd x) {

	// Create a blank p by p matrix
	Eigen::SparseMatrix<double> newX(p, p);

	// For each vector element, multiply by the corresponding A
	for (int i=0; i<n; i++) {
		newX += As[i] * x(i);
	}

	// Return this new matrix
	return newX;

}

// Objective function
double f(Eigen::SparseMatrix<double> XZero) {

	// Init the return val
	double val = 0.0;

	// For each pair of measurements
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
	
			// For each outcome of these measurements
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// Start locations of the real and imag submatrices
					int r1 = (i*numOutcomeB + k) * d;
					int r2 = (j*numOutcomeB + l) * d;

					// This is very weird unless each block is given its own memory
					Eigen::SparseMatrix<double> XReal1 = XZero.block(r1,r1,d,d);
					Eigen::SparseMatrix<double> XReal2 = XZero.block(r2,r2,d,d);
					Eigen::SparseMatrix<double> XImag1 = XZero.block(r1+halfP,r1,d,d);
					Eigen::SparseMatrix<double> XImag2 = XZero.block(r2,r2+halfP,d,d);

					// For readability
					double prodReal = XReal1.cwiseProduct(XReal2).sum();
					double prodImag = XImag1.cwiseProduct(XImag2).sum();
					double den = 1.0 - prodReal + prodImag;

					// Update the value
					val -= std::sqrt(den);

				}
			}

		}
	}

	// Return the function value
	return fScaling*val;

}

// Gradient of the objective function
Eigen::VectorXd delf(Eigen::SparseMatrix<double> XZero) {

	// Init the return val
	Eigen::VectorXd vals = Eigen::VectorXd::Zero(n);

	// For each pair of measurements
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
	
			// For each outcome of these measurements
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// Start locations of the real and imag submatrices
					int r1 = (i*numOutcomeB + k) * d;
					int r2 = (j*numOutcomeB + l) * d;

					// Extract the blocks for this i, j
					Eigen::SparseMatrix<double> XReal1 = XZero.block(r1,r1,d,d);
					Eigen::SparseMatrix<double> XReal2 = XZero.block(r2,r2,d,d);
					Eigen::SparseMatrix<double> XImag1 = XZero.block(r1+halfP,r1,d,d);
					Eigen::SparseMatrix<double> XImag2 = XZero.block(r2,r2+halfP,d,d);

					// For readability
					double prodReal = XReal1.cwiseProduct(XReal2).sum();
					double prodImag = XImag1.cwiseProduct(XImag2).sum();
					double den = 1.0 - prodReal + prodImag;

					// For each component of the vector
					for (int b=0; b<n; b++) {

						// Extract the blocks for this b
						Eigen::SparseMatrix<double> BReal1 = As[b].block(r1,r1,d,d);
						Eigen::SparseMatrix<double> BReal2 = As[b].block(r2,r2,d,d);
						Eigen::SparseMatrix<double> BImag1 = As[b].block(r1+halfP,r1,d,d);
						Eigen::SparseMatrix<double> BImag2 = As[b].block(r2,r2+halfP,d,d);

						// Components with As[b] and X
						double LRBX = BReal1.cwiseProduct(XReal2).sum();
						double RRBX = XReal1.cwiseProduct(BReal2).sum();
						double LIBX = BImag1.cwiseProduct(XImag2).sum();
						double RIBX = XImag1.cwiseProduct(BImag2).sum();

						// Add this inner product of the submatrices
						vals(b) -= 0.5 * std::pow(den, -0.5) * (-LRBX-RRBX+LIBX+RIBX);

					}
				}

			}
		}

	}

	// Return the function value
	return fScaling*vals;

}

// Double differential of the objective function TODO 80+% for bigger problems
Eigen::MatrixXd del2f(Eigen::SparseMatrix<double> XZero) {

	// Create an n by n matrix of all zeros
	Eigen::MatrixXd vals = Eigen::MatrixXd::Zero(n, n);

	// For each pair of measurements
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
	
			// For each outcome of these measurements
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// Start locations of the real and imag submatrices
					int r1 = (i*numOutcomeB + k) * d;
					int r2 = (j*numOutcomeB + l) * d;

					// Extract the blocks for this i, j
					Eigen::SparseMatrix<double> XReal1 = XZero.block(r1,r1,d,d);
					Eigen::SparseMatrix<double> XReal2 = XZero.block(r2,r2,d,d);
					Eigen::SparseMatrix<double> XImag1 = XZero.block(r1+halfP,r1,d,d);
					Eigen::SparseMatrix<double> XImag2 = XZero.block(r2,r2+halfP,d,d);

					// The original value inside the square root
					double prodReal = XReal1.cwiseProduct(XReal2).sum();
					double prodImag = XImag1.cwiseProduct(XImag2).sum();
					double den = 1.0 - prodReal + prodImag;

					// For each component of the vector
					#pragma omp parallel for
					for (int b=0; b<n; b++) {

						// Extract the blocks for this b
						Eigen::SparseMatrix<double> BReal1 = As[b].block(r1,r1,d,d);
						Eigen::SparseMatrix<double> BReal2 = As[b].block(r2,r2,d,d);
						Eigen::SparseMatrix<double> BImag1 = As[b].block(r1+halfP,r1,d,d);
						Eigen::SparseMatrix<double> BImag2 = As[b].block(r2,r2+halfP,d,d);

						// Components with As[b] and X
						double LRBX = BReal1.cwiseProduct(XReal2).sum();
						double RRBX = XReal1.cwiseProduct(BReal2).sum();
						double LIBX = BImag1.cwiseProduct(XImag2).sum();
						double RIBX = XImag1.cwiseProduct(BImag2).sum();

						// For each component of the vector
						for (int c=0; c<n; c++) {

							// Extract the blocks for this c
							Eigen::SparseMatrix<double> CReal1 = As[c].block(r1,r1,d,d);
							Eigen::SparseMatrix<double> CReal2 = As[c].block(r2,r2,d,d);
							Eigen::SparseMatrix<double> CImag1 = As[c].block(r1+halfP,r1,d,d);
							Eigen::SparseMatrix<double> CImag2 = As[c].block(r2,r2+halfP,d,d);

							// Components with As[c] and X
							double LRCX = CReal1.cwiseProduct(XReal2).sum();
							double RRCX = XReal1.cwiseProduct(CReal2).sum();
							double LICX = CImag1.cwiseProduct(XImag2).sum();
							double RICX = XImag1.cwiseProduct(CImag2).sum();

							// Components with As[b] and As[c]
							double LRBC = BReal1.cwiseProduct(CReal2).sum();
							double RRBC = CReal1.cwiseProduct(BReal2).sum();
							double LIBC = BImag1.cwiseProduct(CImag2).sum();
							double RIBC = CImag1.cwiseProduct(BImag2).sum();

							// Add this inner product of the submatrices
							vals(b, c) += 0.25 * std::pow(den, -1.5) * (-LRBX-RRBX+LIBX+RIBX) * (-LRCX-RRCX+LICX+RICX) - 0.5 * std::pow(den, -0.5) * (-LRBC-RRBC+LIBC+RIBC);

						}
					}

				}
			}

		}
	}

	// Return the matrix
	return fScaling*vals;

}

// Constraint function
Eigen::VectorXd g(Eigen::SparseMatrix<double> XZero) {

	// Vector to create
	Eigen::VectorXd gOutput(m);

	// Force the measurement to be projective
	gOutput(0) = (XZero*XZero - XZero).squaredNorm();

	// Return this vector of things that should be zero
	return gScaling*gOutput;

}

// Gradient of the constraint function
Eigen::MatrixXd delg(Eigen::SparseMatrix<double> XZero) {

	// Matrix representing the Jacobian of g
	Eigen::MatrixXd gOutput(m, n);

	// Cache X^2-X
	Eigen::SparseMatrix<double> GCached = XZero*XZero - XZero;

	// The derivative wrt each x
	for (int i=0; i<n; i++) {
		gOutput(0, i) = 2*GCached.cwiseProduct(2*As[i]*XZero - As[i]).sum();
	}

	// Return the gradient vector of g
	return gScaling*gOutput;

}

// Second derivatives of the constraint function if m=1 TODO 10%
Eigen::MatrixXd del2g(Eigen::SparseMatrix<double> XZero) {

	// Matrix representing the Jacobian of g (should really be n x n x m)
	Eigen::MatrixXd gOutput(n, n);

	// Cache X^2-X
	Eigen::SparseMatrix<double> GCached = XZero*XZero - XZero;

	// The derivative wrt each x
    #pragma omp parallel for
	for (int i=0; i<n; i++) {
		Eigen::SparseMatrix<double> AsiXZero = 2*As[i]*XZero - As[i];
		for (int j=0; j<n; j++) {
			gOutput(i, j) = 2*((2*As[j]*XZero - As[j]).cwiseProduct(AsiXZero).sum()) + 4*(GCached.cwiseProduct(As[i]*As[j]).sum());
		}
	}

	// Return the gradient vector of g, scaled by an arbitrary factor
	return gScaling*gOutput;

}

// The Lagrangian 
double L(Eigen::SparseMatrix<double> XZero, Eigen::SparseMatrix<double> XCached, Eigen::VectorXd y, Eigen::MatrixXd Z) {
	return f(XZero) - y.transpose()*g(XZero) - XCached.cwiseProduct(Z).sum();
}

// Differential of the Lagrangian given individual components
Eigen::VectorXd delL(Eigen::VectorXd y, Eigen::MatrixXd Z, Eigen::VectorXd delfCached, Eigen::MatrixXd A_0) {

	// Calculate A* Z
	Eigen::VectorXd AStarZ = Eigen::VectorXd::Zero(n);
	for (int i=0; i<n; i++) {
		AStarZ(i) = As[i].cwiseProduct(Z).sum();
	}

	// Return this vector
	return delfCached - A_0.transpose()*y - AStarZ;

}

// Double differential of the Lagrangian given an interior point
Eigen::MatrixXd del2L(Eigen::SparseMatrix<double> XZero, Eigen::VectorXd y) {

	// In our case the second derivative of the A dot Z is zero
	return del2f(XZero) - del2g(XZero)*y(0);

}

// Function giving the norm of a point, modified by some mu
double rMag(double mu, Eigen::MatrixXd Z, Eigen::SparseMatrix<double> XCached, Eigen::VectorXd delLCached, Eigen::VectorXd gCached) {

	// The left part of the square root
	Eigen::VectorXd left = Eigen::VectorXd::Zero(n+m);
	left << delLCached, gCached;

	// The right part of the square root
	Eigen::MatrixXd right = XCached * Z - mu * Eigen::MatrixXd::Identity(p,p);

	// Sum the l^2/Frobenius norms
	double val = std::sqrt(left.squaredNorm() + right.squaredNorm());

	// Return this magnitude
	return val;

}

// The merit function
double F(Eigen::VectorXd x, Eigen::SparseMatrix<double> ZSparse, double mu) {

	// Cache the X matrix
	Eigen::SparseMatrix<double> XCached = X(x);
	Eigen::SparseMatrix<double> XZero = X(x, 0.0);
	Eigen::VectorXd gCached = g(XZero);
	double XDeter = Eigen::MatrixXd(XCached).determinant();
	double ZDeter = Eigen::MatrixXd(ZSparse).determinant();

	// Calculate the two components
	double FBP = f(XZero) - mu*std::log(XDeter) + rho*gCached.norm();
	double FPD = XCached.cwiseProduct(ZSparse).sum() - mu*std::log(XDeter*ZDeter);

	// Return the sum
	return FBP + nu*FPD;

}

// The change in merit function
double deltaF(Eigen::MatrixXd deltaZ, Eigen::SparseMatrix<double> ZInverse, Eigen::SparseMatrix<double> ZSparse, Eigen::SparseMatrix<double> XCached, Eigen::SparseMatrix<double> XInverse, Eigen::VectorXd delfCached, Eigen::VectorXd gCached, Eigen::MatrixXd A_0, Eigen::VectorXd deltax) {

	// Calculate the deltaX matrix
	Eigen::MatrixXd deltaX = XNoB(deltax);

	// Calculate the two components
	double FBP = delfCached.dot(deltax) - mu*(XInverse*deltaX).trace() + rho*((gCached+A_0*deltax).norm()-gCached.norm());
	double FPD = (deltaX*ZSparse + XCached*deltaZ - mu*XInverse*deltaX - mu*ZInverse*deltaZ).trace();

	// Return the sum
	return FBP + nu*FPD;

}

// Returns true if a matrix can be Cholesky decomposed
bool isPD(Eigen::MatrixXd G) {
	return G.ldlt().info() != Eigen::NumericalIssue;
}
bool isComplexPD(Eigen::MatrixXcd G) {
	return G.ldlt().info() != Eigen::NumericalIssue;
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
			std::cout << "                        " << std::endl;
			std::cout << "       output options          " << std::endl;
			std::cout << " -p [int]         set the precision" << std::endl;
			std::cout << " -B               output only the iter count for benchmarking" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "       init options          " << std::endl;
			std::cout << " -R               use a random seed" << std::endl;
			std::cout << " -K               use the ideal if known" << std::endl;
			std::cout << " -Y               use nearby the ideal if known" << std::endl;
			std::cout << " -T [dbl]         set the threshold for the initial g(x)" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "    parameter options          " << std::endl;
			std::cout << " -D [dbl]         set the extra diagonal" << std::endl;
			std::cout << " -e [dbl]         set epsilon" << std::endl;
			std::cout << " -E [dbl]         set epsilon_0" << std::endl;
			std::cout << " -M [dbl]         set M_c" << std::endl;
			std::cout << " -b [dbl]         set beta" << std::endl;
			std::cout << " -N [dbl]         set nu" << std::endl;
			std::cout << " -m [dbl]         set initial mu" << std::endl;
			std::cout << " -s [dbl]         set mu scaling (mu->mu/this)" << std::endl;
			std::cout << " -r [dbl]         set rho" << std::endl;
			std::cout << " -g [dbl]         set gamma" << std::endl;
			std::cout << " -G [dbl]         set the factor of g(x)" << std::endl;
			std::cout << " -F [dbl]         set the factor of f(x)" << std::endl;
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

		// If told to use the known exact solution
		} else if (arg == "-K") {
			initMode = "exact";

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

		// Set the f(x) scaling
		} else if (arg == "-F") {
			fScaling = std::stod(argv[i+1]);
			i += 1;

		// Set the threshold for the initial g(x)
		} else if (arg == "-T") {
			gThresh = std::stod(argv[i+1]);
			i += 1;

		// Set the g(x) scaling
		} else if (arg == "-G") {
			gScaling = std::stod(argv[i+1]);
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

	// Output formatting
	std::cout << std::setprecision(precision);

	// Start the timer 
	auto t1 = std::chrono::high_resolution_clock::now();

	// Useful quantities
	numPerm = sets*(sets-1)/2;
	numMeasureB = sets;
	numOutcomeB = d;
	numUniquePer = (d*(d+1))/2-1;
	numRealPer = numUniquePer;
	numImagPer = numUniquePer+1-d;
	numTotalPer = numRealPer + numImagPer;

	// Sizes of matrices
	n = numMeasureB*(numOutcomeB-1)*numTotalPer;
	m = 1;
	p = numMeasureB*numOutcomeB*d*2;
	halfP = p / 2;

	// Output various bits of info about the problem/parameters
	if (outputMode == "") {
		std::cout << "--------------------------------" << std::endl;
		std::cout << "          System Info           " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "               d = " << d << std::endl;
		std::cout << "            sets = " << sets << std::endl;
		std::cout << "  size of vector = " << n << std::endl;
		std::cout << "  size of matrix = " << p << " x " << p << std::endl;
		std::cout << "                 ~ " << p*p*16 / (1024*1024) << " MB "<< std::endl;
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
		std::cout << "     f(x) factor = " << fScaling << std::endl;
		std::cout << "     g(x) factor = " << gScaling << std::endl;
		std::cout << "" << std::endl;
	}

	// The "ideal" value
	double maxVal = fScaling*numPerm*d*std::sqrt(d*(d-1));

	// Calculate the A matrices uses to turn X to x
	for (int i=0; i<numMeasureB; i++) {
		for (int k=0; k<numOutcomeB-1; k++) {

			// Where in X/x this matrix starts
			int matLoc = (i*numOutcomeB + k) * d;
			int finalMatLoc = (i*numOutcomeB + numOutcomeB-1) * d;
			int vecLocReal = (i*(numOutcomeB-1) + k) * numTotalPer;
			int vecLocImag = vecLocReal + numRealPer;
			int nextX = 0;
			int nextY = 0;

			// For each real vector element in that matrix
			for (int l=0; l<numRealPer; l++) {

				// Create a blank p by p matrix
				Eigen::SparseMatrix<double> newAReal(p, p);
				std::vector<Eigen::Triplet<double>> trips;

				// For the diagonals
				if (nextX == nextY) {

					// Place the real comps in the diagonal blocks
					trips.push_back(Eigen::Triplet<double>(matLoc+nextX, matLoc+nextY, 1));
					trips.push_back(Eigen::Triplet<double>(matLoc+nextX+halfP, matLoc+nextY+halfP, 1));

					// Subtract to force trace one
					trips.push_back(Eigen::Triplet<double>(matLoc+d-1, matLoc+d-1, -1));
					trips.push_back(Eigen::Triplet<double>(matLoc+d-1+halfP, matLoc+d-1+halfP, -1));
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+d-1, finalMatLoc+d-1, 1));
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+halfP+d-1, finalMatLoc+halfP+d-1, 1));

					// Subtract to force sum to identity
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextX, finalMatLoc+nextY, -1));
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextX+halfP, finalMatLoc+nextY+halfP, -1));

				// For the non-diagonals
				} else {

					// Place the real comps in the diagonal blocks
					trips.push_back(Eigen::Triplet<double>(matLoc+nextX, matLoc+nextY, 1));
					trips.push_back(Eigen::Triplet<double>(matLoc+nextY, matLoc+nextX, 1));
					trips.push_back(Eigen::Triplet<double>(matLoc+nextX+halfP, matLoc+nextY+halfP, 1));
					trips.push_back(Eigen::Triplet<double>(matLoc+nextY+halfP, matLoc+nextX+halfP, 1));

					// Subtract to force sum to identity
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextX, finalMatLoc+nextY, -1));
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextY, finalMatLoc+nextX, -1));
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextX+halfP, finalMatLoc+nextY+halfP, -1));
					trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextY+halfP, finalMatLoc+nextX+halfP, -1));

				}
				
				// Move the location along
				nextX++;
				if (nextX >= d) {
					nextY++;
					nextX = nextY;
				}

				// Add this matrix to the list
				newAReal.setFromTriplets(trips.begin(), trips.end());
				As.push_back(newAReal);

			}

			// For each imag vector element
			nextX = 1;
			nextY = 0;
			for (int l=0; l<numImagPer; l++) {

				// Create a blank p by p matrix
				//Eigen::MatrixXd newAImag = Eigen::MatrixXd::Zero(p, p);

				// Create a blank p by p matrix
				Eigen::SparseMatrix<double> newAImag(p, p);
				std::vector<Eigen::Triplet<double>> trips;

				// Place the imag comps in the off-diagonal blocks
				trips.push_back(Eigen::Triplet<double>(matLoc+nextX+halfP, matLoc+nextY, 1));
				trips.push_back(Eigen::Triplet<double>(matLoc+nextY, matLoc+nextX+halfP, 1));
				trips.push_back(Eigen::Triplet<double>(matLoc+nextX, matLoc+nextY+halfP, -1));
				trips.push_back(Eigen::Triplet<double>(matLoc+nextY+halfP, matLoc+nextX, -1));

				// Subtract to force sum to identity
				trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextX+halfP, finalMatLoc+nextY, -1));
				trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextY, finalMatLoc+nextX+halfP, -1));
				trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextX, finalMatLoc+nextY+halfP, 1));
				trips.push_back(Eigen::Triplet<double>(finalMatLoc+nextY+halfP, finalMatLoc+nextX, 1));

				// Move the location along
				nextX++;
				if (nextX >= d) {
					nextY++;
					nextX = nextY+1;
				}

				// Add this matrix to the list
				newAImag.setFromTriplets(trips.begin(), trips.end());
				As.push_back(newAImag);

			}

		}

	}

	// Calculate the B matrix
	B = Eigen::SparseMatrix<double>(p, p);
	std::vector<Eigen::Triplet<double>> tripsB;
	for (int i=0; i<numMeasureB; i++) {
		for (int k=0; k<numOutcomeB-1; k++) {

			// The trace of each should be 1
			int matLoc = (i*numOutcomeB + k) * d;
			tripsB.push_back(Eigen::Triplet<double>(matLoc+d-1, matLoc+d-1, 1));
			tripsB.push_back(Eigen::Triplet<double>(matLoc+halfP+d-1, matLoc+halfP+d-1, 1));

		}

		// The last of each measurement should be the identity
		int matLoc = (i*numOutcomeB + numOutcomeB-1) * d;
		for (int a=0; a<d-1; a++) {
			tripsB.push_back(Eigen::Triplet<double>(matLoc+a, matLoc+a, 1));
			tripsB.push_back(Eigen::Triplet<double>(matLoc+halfP+a, matLoc+halfP+a, 1));
		}
		tripsB.push_back(Eigen::Triplet<double>(matLoc+d-1, matLoc+d-1, 2-d));
		tripsB.push_back(Eigen::Triplet<double>(matLoc+halfP+d-1, matLoc+halfP+d-1, 2-d));

	}
	B.setFromTriplets(tripsB.begin(), tripsB.end());

	// The interior point to optimise
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
	Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(p, p);
	Eigen::SparseMatrix<double> ZSparse(p, p);

	// The (currently blank) cached form of X(x)
	Eigen::SparseMatrix<double> XZero(p, p);
	Eigen::SparseMatrix<double> XCached(p, p);
	Eigen::MatrixXd XDense = Eigen::MatrixXd::Zero(p, p);

	// Seed so it's random each time
	if (initMode == "random") {
		srand((unsigned int) time(0));
	}

	// If starting with a random matrix (seeded or not)
	if (initMode == "random" || initMode == "fixed") {

		// Start with a bunch of projective measurements
		for (int i=0; i<numMeasureB; i++) {
			for (int j=0; j<numOutcomeB; j++) {

				// Start with a random normalized vector
				Eigen::VectorXcd vec = Eigen::VectorXcd::Random(d).normalized();

				// Turn it into a projective measurement
				Eigen::MatrixXcd tempMat = vec * vec.adjoint();

				// Copy it to the big matrix
				int ind = (i*numOutcomeB + j) * d;
				XDense.block(ind, ind, d, d) = tempMat.real();
				XDense.block(ind, ind+halfP, d, d) = tempMat.imag();

			}
		}

		// Extract the x from this X
		x = Xtox(XDense);

		// If output is allowed
		if (outputMode == "") {
			std::cout << "--------------------------------" << std::endl;
			std::cout << "     Finding Interior Point " << std::endl;;
			std::cout << "--------------------------------" << std::endl;
		}

		// Gradient descent to make sure we start with an interior point
		double v = 0;
		for (int i=0; i<10000000; i++) {
			XZero = X(x, 0.0);
			x += -delg(XZero).row(0);
			v = g(XZero)(0) / gScaling;
			if (outputMode == "") {
				std::cout << "g(x) = " << v << std::endl;
			}
			if (v < gThresh) {
				break;
			}
		}

	// Use optimum 
	} else if (initMode == "exact") {
		if (sets == 2 && d == 2) {
			x << 1.0, 0.0,   0.0,     
				 0.5, 0.5,   0.0;
		} else if (sets == 3 && d == 2) {
			x << 1.0, 0.0,   0.0,     
				 0.5, 0.5,   0.0, 
				 0.5, 0.0,   0.5;
		} else {
			std::cerr << "Don't know the exact solution for this problem" << std::endl;
			return 0;

		}

	// Use nearby the optimum TODO copy results from seesaw
	} else if (initMode == "nearby") {
		if (d== 2 && sets == 2) {
			double v = 0.4;
			x << 1.0, 0.0,   0.0,     
				 v, std::sqrt(v*(1-v)),   0.0;
		} else if (d == 2 && sets == 3) {
			double v = 0.4;
			x << 1.0, 0.0,   0.0,     
				 v, std::sqrt(v*(1-v)),   0.0, 
				 0.5, 0.0,   0.5;
		} else {
			std::cerr << "Don't have a nearby point for this problem" << std::endl;
			return 0;

		}

	}

	// Get the full matrices from this
	XZero = X(x, 0.0);
	XCached = X(x);
	XDense = Eigen::MatrixXd(XCached);

	// Output the initial X
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "        Initial Matrices        " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		Eigen::MatrixXcd M(d, d);
		for (int i=0; i<numMeasureB; i++) {
			for (int j=0; j<numOutcomeB; j++) {
				int ind = (i*numOutcomeB + j)*d;
				M = Eigen::MatrixXd(XZero.block(ind, ind, d, d)) + 1i*Eigen::MatrixXd(XZero.block(ind+halfP, ind, d, d));
				std::cout << std::endl;
				prettyPrint("M_" + std::to_string(j) + "^" + std::to_string(i) + " = ", M);
				std::cout << std::endl;
				std::cout << "|M^2-M|  = " << (M.adjoint()*M - M).squaredNorm() << std::endl;
				std::cout << "is M PD? = " << isComplexPD(M) << std::endl;
			}
		}
	}

	// Ensure this is an interior point
	if (g(XZero).squaredNorm() > epsilon/100) {
		std::cerr << "Error - X should start as an interior point (g(x) = " << g(XZero).squaredNorm() << " > epsilon)" << std::endl;
		return 1;
	}
	if (!isPD(XCached)) {
		std::cerr << "Error - X should start as an interior point (X is not semidefinite)" << std::endl;
		return 1;
	}

	// Initialise Z
	Z = Eigen::MatrixXd::Zero(p, p);
	for (int i=0; i<numMeasureB; i++) {
		for (int j=0; j<numOutcomeB; j++) {
			int currentLoc = (i*numOutcomeB + j) * d;
			int copyLoc = (i*numOutcomeB + ((j+1) % numOutcomeB)) * d;
			Z.block(currentLoc,currentLoc,d,d) = XCached.block(copyLoc,copyLoc,d,d);
			Z.block(currentLoc+halfP,currentLoc+halfP,d,d) = XCached.block(copyLoc,copyLoc,d,d);
			Z.block(currentLoc+halfP,currentLoc,d,d) = XCached.block(copyLoc+halfP,copyLoc,d,d);
			Z.block(currentLoc,currentLoc+halfP,d,d) = -XCached.block(copyLoc+halfP,copyLoc,d,d);
		}
	}

	// Output the initial Z
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "      Initial Z " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		for (int i=0; i<numMeasureB; i++) {
			for (int j=0; j<numOutcomeB; j++) {
				int ind = (i*numOutcomeB + j)*d;
				Eigen::MatrixXcd M = Z.block(ind, ind, d, d) + 1i*Z.block(ind+halfP, ind, d, d);
				std::cout << std::endl;
				prettyPrint("Z_" + std::to_string(j) + "^" + std::to_string(i) + " = ", M);
			}
		}
		std::cout << std::endl;
		prettyPrint("X dot Z = ", XCached.cwiseProduct(Z).sum());
	}

	// Init some thing that are used for the first calcs
	Eigen::SparseMatrix<double> XInverse = XDense.inverse().sparseView();
	Eigen::SparseMatrix<double> ZInverse = Z.inverse().sparseView();
	Eigen::MatrixXd gCached = g(XZero);
	Eigen::MatrixXd A_0 = delg(XZero);
	Eigen::VectorXd delfCached = delf(XZero);
	Eigen::VectorXd delLCached = delL(y, Z, delfCached, A_0);
	double rMagZero = rMag(0, Z, XCached, delLCached, gCached);

	// To prevent reinit each time
	Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd AStarXInverse = Eigen::VectorXd::Zero(n);
	Eigen::MatrixXd GHInverse = Eigen::MatrixXd::Zero(n, n);
	Eigen::MatrixXd deltaX = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd deltax = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd deltay = Eigen::VectorXd::Zero(m);
	Eigen::MatrixXd deltaZ = Eigen::MatrixXd::Zero(p, p);

	// Outer loop
	double rMagMu = 0;
	int k = 0;
	int totalInner = 0;
	for (k=0; k<maxOuterIter; k++) {

		// Check if global convergence is reached
		rMagZero = rMag(0, Z, XCached, delLCached, gCached);
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
			std::cout << "     r(w,0) = " << rMagZero << " ?< " << epsilon << std::endl;;
			std::cout << "      -f(x) = " << -f(XZero)/fScaling << " <= " << maxVal << std::endl;;
			std::cout << "       g(x) = " << gCached/gScaling << std::endl;;
			std::cout << "      -L(x) = " << -L(XZero, XCached, y, Z) << std::endl;;
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
			XCached = X(x);
			XZero = X(x, 0.0);
			XDense = Eigen::MatrixXd(XCached);
			ZSparse = Z.sparseView();
			XInverse = XDense.inverse().sparseView();
			ZInverse = Z.inverse().sparseView();
			gCached = g(XZero);
			A_0 = delg(XZero);
			delfCached = delf(XZero);
			delLCached = delL(y, Z, delfCached, A_0);

			// Update G
			G = del2L(XZero, y);

			// See if G is already PD
			if (!isPD(G)) {

				// If G-sigma*I is PD
				double sigma = 1;
				Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
				if (isPD(G-sigma*I)) {

					// Decrease sigma until it isn't
					while (isPD(G-sigma*I) && sigma >= 1e-7) {
						sigma /= 2;
					}

					// Then return to the one that was still PD
					sigma *= 2;

				// If G-sigma*I is not PD
				} else {

					// Increase sigma until it is
					while (!isPD(G-sigma*I)) {
						sigma *= 2;
					}

				}

				// Update this new G
				G = G-sigma*I;

			}	

			// Construct H
			for (int j=0; j<n; j++) {
				Eigen::SparseMatrix<double> cached = XInverse*As[j]*ZSparse;
				for (int i=0; i<n; i++) {
					H(i,j) = As[i].cwiseProduct(cached).sum();
				}
			}

			// Calculate/cache a few useful matrices 
			GHInverse = (G + H).inverse();
			for (int i=0; i<n; i++) {
				AStarXInverse(i) = As[i].cwiseProduct(XInverse).sum();
			}

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
			deltaX = XNoB(deltax);
			deltaZ = mu*XInverse - Z - 0.5*(XInverse*deltaX*Z + Z*deltaX*XInverse);

			// Determine the max l such that beta^l = 1e-9
			int maxL = std::log(epsilon) / std::log(beta);

			// Get a base step size using the min eigenvalues
			double alphaBarX = -gammaVal / std::real((XInverse * deltaX).eigenvalues()[0]);
			double alphaBarZ = -gammaVal / std::real((ZInverse * deltaZ).eigenvalues()[0]);
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
			double FCached = F(x, ZSparse, mu);
			double deltaFCached = deltaF(deltaZ, ZInverse, ZSparse, XCached, XInverse, delfCached, gCached, A_0, deltax);
			for (l=0; l<maxL; l++){
				alpha = alphaBar * std::pow(beta, l);
				if (F(x+alpha*deltax, ZSparse+alpha*deltaZ, mu) <= FCached + epsilonZero*alpha*deltaFCached && isPD(X(x+alpha*deltax))) {
					break;
				}
			}

			// Update variables
			x += alpha*deltax;
			y += deltay;
			Z += alpha*deltaZ;
			
			// Inner-iteration output
			rMagMu = rMag(mu, Z, XCached, delLCached, gCached);
			if (outputMode == "") {
				std::cout << "f = " << f(XZero) << "   r = " << rMagMu  << " ?< " << epsilonPrime << "   g = " << gCached/gScaling << std::endl;
			}

			// Check if local convergence is reached
			if (rMagMu <= epsilonPrime) {
				break;
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

	// Extract the solution from X
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "         Final Matrices " << std::endl;;
		std::cout << "----------------------------------" << std::endl;
		XZero = X(x, 0);
		Eigen::MatrixXcd M(d, d);
		for (int i=0; i<numMeasureB; i++) {
			for (int j=0; j<numOutcomeB; j++) {
				int ind = (i*numOutcomeB + j)*d;
				M = Eigen::MatrixXd(XZero.block(ind, ind, d, d)) + 1i*Eigen::MatrixXd(XZero.block(ind+halfP, ind, d, d));
				std::cout << std::endl;
				prettyPrint("M_" + std::to_string(j) + "^" + std::to_string(i) + " = ", M);
				std::cout << std::endl;
				std::cout << "|M^2-M|  = " << (M.adjoint()*M - M).squaredNorm() << std::endl;
				std::cout << "is M PD? = " << isComplexPD(M) << std::endl;
			}
		}
	}

	// Final output
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "         Final Output " << std::endl;;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "          r(w) = " << rMagZero << " < " << epsilon << std::endl;;
		std::cout << "         -f(x) = " << -f(XZero)/fScaling << " <= " << maxVal << std::endl;;
		std::cout << "          g(x) = " << gCached/gScaling << std::endl;;
		std::cout << "         -L(w) = " << -L(XZero, XCached, y, Z) << std::endl;;
		std::cout << "       delf(w) = " << delfCached.norm()/fScaling << std::endl;;
		std::cout << "       delL(w) = " << delLCached.norm() << std::endl;;
		std::cout << "       delg(w) = " << A_0.norm()/gScaling << std::endl;;
		std::cout << "   outer iters = " << k << std::endl;;
		std::cout << "   total inner = " << totalInner << std::endl;;
		std::cout << "          time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;

	// Benchmarking mode
	} else if (outputMode == "B") {
		std::cout << totalInner << std::endl;;

	}

	// Everything went fine
	return 0;

}

