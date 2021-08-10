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

// Optimisation parameters
double extraDiag = 0.0001;
int maxOuterIter = 10000;
int maxInnerIter = 10000;
double muScaling = 10;

// Parameters between 0 and 1
double gammaVal = 0.1;
double epsilonZero = 0.9;
double beta = 0.7;

// Parameters greater than 0
double epsilon = 0.01;   // The convergence threshold for the outer
double M_c = 1000000;    // M_c * mu is the convergence threshold for the inner
double mu = 1.0;         // This decreases each iteration and affects many things
double nu = 0.9;         // Affects the merit function
double rho = 0.5;        // Affects the merit function

// Useful quantities
int numPerm = 0;
int numMeasureB = 0;
int numOutcomeB = 0;
int numUniquePer = 0;
int numRealPer = 0;
int numImagPer = 0;
int numTotalPer = 0;
std::vector<Eigen::MatrixXd> As;
Eigen::MatrixXd B;

// Sizes of matrices
int n = 0;
int m = 0;
int p = 0;

// For printing
int precision = 5;

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

// Function turning x to X
Eigen::MatrixXd X(Eigen::VectorXd x, double extra=extraDiag) {

	// Create a blank p by p matrix
	Eigen::MatrixXd newX = Eigen::MatrixXd::Zero(p, p);

	// For each vector element, multiply by the corresponding A
	for (int i=0; i<n; i++) {
		newX += As[i] * x(i);
	}

	// Add the B
	newX += B;

	// Add a bit extra to make it reversible
	newX += Eigen::MatrixXd::Identity(p, p) * extra;
	
	// Return this new matrix
	return newX;

}

// Function turning x to X without any B addition
Eigen::MatrixXd XNoB(Eigen::VectorXd x) {

	// Create a blank p by p matrix
	Eigen::MatrixXd newX = Eigen::MatrixXd::Zero(p, p);

	// For each vector element, multiply by the corresponding A
	for (int i=0; i<n; i++) {
		newX += As[i] * x(i);
	}

	// Return this new matrix
	return newX;

}

// Objective function
double f(Eigen::VectorXd x) {

	// Init the return val
	double val = 0.0;

	// Cache the full matrix
	Eigen::MatrixXd XCached = X(x, 0.0);
	int halfP = p / 2;

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
					Eigen::MatrixXd XReal1 = XCached.block(r1,r1,d,d);
					Eigen::MatrixXd XReal2 = XCached.block(r2,r2,d,d);
					Eigen::MatrixXd XImag1 = XCached.block(r1+halfP,r1,d,d);
					Eigen::MatrixXd XImag2 = XCached.block(r2,r2+halfP,d,d);

					// For readability
					double prodReal = XReal1.cwiseProduct(XReal2).sum();
					double prodImag = XImag1.cwiseProduct(XImag2).sum();
					double d = 1.0 - prodReal + prodImag;

					// Update the value
					val -= std::sqrt(d);

				}
			}

		}
	}

	// Return the function value
	return val;

}

// Gradient of the objective function
Eigen::VectorXd delf(Eigen::VectorXd x) {

	// Init the return val
	Eigen::VectorXd vals = Eigen::VectorXd::Zero(n);

	// Cache the full matrix
	Eigen::MatrixXd XCached = X(x, 0.0);
	int halfP = p / 2;

	// For each component of the vector
	for (int b=0; b<n; b++) {

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
						Eigen::MatrixXd XReal1 = XCached.block(r1,r1,d,d);
						Eigen::MatrixXd XReal2 = XCached.block(r2,r2,d,d);
						Eigen::MatrixXd XImag1 = XCached.block(r1+halfP,r1,d,d);
						Eigen::MatrixXd XImag2 = XCached.block(r2,r2+halfP,d,d);
						Eigen::MatrixXd BReal1 = As[b].block(r1,r1,d,d);
						Eigen::MatrixXd BReal2 = As[b].block(r2,r2,d,d);
						Eigen::MatrixXd BImag1 = As[b].block(r1+halfP,r1,d,d);
						Eigen::MatrixXd BImag2 = As[b].block(r2,r2+halfP,d,d);

						// For readability
						double LRBX = BReal1.cwiseProduct(XReal2).sum();
						double RRBX = XReal1.cwiseProduct(BReal2).sum();
						double LIBX = BImag1.cwiseProduct(XImag2).sum();
						double RIBX = XImag1.cwiseProduct(BImag2).sum();
						double prodReal = XReal1.cwiseProduct(XReal2).sum();
						double prodImag = XImag1.cwiseProduct(XImag2).sum();
						double d = 1.0 - prodReal + prodImag;

						// Add this inner product of the submatrices
						vals(b) -= 0.5 * std::pow(d, -0.5) * (-LRBX-RRBX+LIBX+RIBX);

					}
				}

			}
		}

	}

	// Return the function value
	return vals;

}

// Double differential of the objective function
Eigen::MatrixXd del2f(Eigen::VectorXd x) {

	// Create an n by n matrix of all zeros
	Eigen::MatrixXd vals = Eigen::MatrixXd::Zero(n, n);

	// Cache the full matrix
	Eigen::MatrixXd XCached = X(x, 0.0);
	int halfP = p / 2;

	// For each component of the vector
	for (int b=0; b<n; b++) {
		for (int c=0; c<n; c++) {

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
							Eigen::MatrixXd XReal1 = XCached.block(r1,r1,d,d);
							Eigen::MatrixXd XReal2 = XCached.block(r2,r2,d,d);
							Eigen::MatrixXd XImag1 = XCached.block(r1+halfP,r1,d,d);
							Eigen::MatrixXd XImag2 = XCached.block(r2,r2+halfP,d,d);
							Eigen::MatrixXd BReal1 = As[b].block(r1,r1,d,d);
							Eigen::MatrixXd BReal2 = As[b].block(r2,r2,d,d);
							Eigen::MatrixXd BImag1 = As[b].block(r1+halfP,r1,d,d);
							Eigen::MatrixXd BImag2 = As[b].block(r2,r2+halfP,d,d);
							Eigen::MatrixXd CReal1 = As[c].block(r1,r1,d,d);
							Eigen::MatrixXd CReal2 = As[c].block(r2,r2,d,d);
							Eigen::MatrixXd CImag1 = As[c].block(r1+halfP,r1,d,d);
							Eigen::MatrixXd CImag2 = As[c].block(r2,r2+halfP,d,d);

							// Components with As[b] and X
							double LRBX = BReal1.cwiseProduct(XReal2).sum();
							double RRBX = XReal1.cwiseProduct(BReal2).sum();
							double LIBX = BImag1.cwiseProduct(XImag2).sum();
							double RIBX = XImag1.cwiseProduct(BImag2).sum();

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

							// For readability
							double prodReal = XReal1.cwiseProduct(XReal2).sum();
							double prodImag = XImag1.cwiseProduct(XImag2).sum();
							double d = 1.0 - prodReal + prodImag;

							// Add this inner product of the submatrices
							vals(b, c) += 0.25 * std::pow(d, -1.5) * (-LRBX-RRBX+LIBX+RIBX) * (-LRCX-RRCX+LICX+RICX) - 0.5 * std::pow(d, -0.5) * (-LRBC-RRBC+LIBC+RIBC);

						}
					}

				}
			}

		}
	}

	// Return the matrix
	return vals;

}

// Constraint function
Eigen::VectorXd g(Eigen::VectorXd x) {

	// Vector to create
	Eigen::VectorXd gOutput = Eigen::VectorXd::Zero(m);

	// Cache the full matrix
	Eigen::MatrixXd XCached = X(x, 0.0);
	int halfP = p / 2;

	// Force the measurement to be projective
	gOutput(0) = XCached.cwiseProduct(XCached).sum() - XCached.trace();

	// Return this vector of things that should be zero
	return gOutput;

}

// Gradient of the constraint function 
Eigen::MatrixXd delg(Eigen::VectorXd x) {

	// Matrix representing the Jacobian of g
	Eigen::MatrixXd gOutput = Eigen::MatrixXd::Zero(m, n);

	// Cache the full matrix
	Eigen::MatrixXd XCached = X(x, 0.0);

	// The derivative wrt each x
	for (int i=0; i<n; i++) {
		gOutput(0, i) = 2*As[i].cwiseProduct(XCached).sum() - As[i].trace();
	}

	// Return the gradient vector of g
	return gOutput;

}

// Second derivatives of the constraint function if m=1
Eigen::MatrixXd del2g(Eigen::VectorXd x) {

	// Matrix representing the Jacobian of g (should really be n x n x m)
	Eigen::MatrixXd gOutput = Eigen::MatrixXd::Zero(n, n);

	// The derivative wrt each x
	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++) {
			gOutput(i, j) = 2*As[i].cwiseProduct(As[j]).sum();
		}
	}

	// Return the gradient vector of g
	return gOutput;

}

// The Lagrangian 
double L(Eigen::VectorXd x, Eigen::VectorXd y, Eigen::MatrixXd Z) {
	return f(x) - y.transpose()*g(x) - X(x).cwiseProduct(Z).sum();
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
Eigen::MatrixXd del2L(Eigen::VectorXd x, Eigen::VectorXd y) {

	// In our case the second derivative of the A dot Z is zero
	return del2f(x) - del2g(x)*y(0);

}

// Function giving the norm of a point, modified by some mu
double rMag(double mu, Eigen::MatrixXd Z, Eigen::MatrixXd XCached, Eigen::VectorXd delLCached, Eigen::VectorXd gCached) {

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
double F(Eigen::VectorXd x, Eigen::MatrixXd Z, double mu) {

	// Cache the X matrix
	Eigen::MatrixXd XCached = X(x);
	Eigen::VectorXd gCached = g(x);

	// Calculate the two components
	double FBP = f(x) - mu*log(XCached.determinant()) + rho*gCached.norm();
	double FPD = XCached.cwiseProduct(Z).sum() - mu*std::log(XCached.determinant()*Z.determinant());

	// Return the sum
	return FBP + nu*FPD;

}

// The change in merit function
double deltaF(Eigen::VectorXd x, Eigen::VectorXd deltax, Eigen::MatrixXd Z, Eigen::MatrixXd deltaZ, double mu) {

	// Cache the X matrix
	Eigen::MatrixXd XCached = X(x);
	Eigen::MatrixXd deltaX = XNoB(deltax);
	Eigen::MatrixXd XInverse = XCached.inverse();
	Eigen::MatrixXd ZInverse = Z.inverse();
	Eigen::VectorXd delfCached = delf(x);
	Eigen::VectorXd gCached = g(x);
	Eigen::MatrixXd A_0 = delg(x);

	// Calculate the two components
	double FBP = delfCached.dot(deltax) - mu*(XInverse*deltaX).trace() + rho*((gCached+A_0*deltax).norm()-gCached.norm());
	double FPD = (deltaX*Z + XCached*deltaZ - mu*XInverse*deltaX - mu*ZInverse*deltaZ).trace();

	// Return the sum
	return FBP + nu*FPD;

}

// Returns true if a matrix can be Cholesky decomposed
bool isPD(Eigen::MatrixXd G) {
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
			std::cout << "    main options          " << std::endl;
			std::cout << " -h               show the help" << std::endl;
			std::cout << " -d [int]         set the dimension" << std::endl;
			std::cout << " -n [int]         set the number of measurements" << std::endl;
			std::cout << " -p [int]         set the precision" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "    tolerance options          " << std::endl;
			std::cout << " -e [dbl]         set epsilon" << std::endl;
			std::cout << " -E [dbl]         set epsilon_0" << std::endl;
			std::cout << " -M [dbl]         set M_c" << std::endl;
			std::cout << " -b [dbl]         set beta" << std::endl;
			std::cout << " -N [dbl]         set nu" << std::endl;
			std::cout << " -m [dbl]         set initial mu" << std::endl;
			std::cout << " -s [dbl]         set mu scaling (mu->mu/this)" << std::endl;
			std::cout << " -r [dbl]         set rho" << std::endl;
			std::cout << " -g [dbl]         set gamma" << std::endl;
			std::cout << " -I [int]         set outer iteration limit" << std::endl;
			std::cout << " -i [int]         set inner iteration limit" << std::endl;
			std::cout << "" << std::endl;
			return 0;

		// Set the number of measurements 
		} else if (arg == "-d") {
			d = std::stoi(argv[i+1]);
			i += 1;

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

		// Set the mu scaling
		} else if (arg == "-s") {
			muScaling = std::stod(argv[i+1]);
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

	std::cout << "numUniquePer = " << numUniquePer << std::endl;
	std::cout << "numRealPer = " << numRealPer << std::endl;
	std::cout << "numImagPer = " << numImagPer << std::endl;
	std::cout << "numTotalPer = " << numTotalPer << std::endl;
	std::cout << "n = " << n << std::endl;
	std::cout << "p = " << p << std::endl;

	// The "ideal" value
	double maxVal = numPerm*d*std::sqrt(d*(d-1));

	// Calculate the A matrices uses to turn X to x
	int halfP = p / 2;
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
				Eigen::MatrixXd newAReal = Eigen::MatrixXd::Zero(p, p);

				// Place the real comps in the diagonal blocks
				newAReal(matLoc+nextX, matLoc+nextY) = 1;
				newAReal(matLoc+nextY, matLoc+nextX) = 1;
				newAReal(matLoc+nextX+halfP, matLoc+nextY+halfP) = 1;
				newAReal(matLoc+nextY+halfP, matLoc+nextX+halfP) = 1;

				// Subtract to force trace one
				if (nextX == nextY) {
					newAReal(matLoc+d-1, matLoc+d-1) = -1;
					newAReal(matLoc+d-1, matLoc+d-1) = -1;
					newAReal(matLoc+d-1+halfP, matLoc+d-1+halfP) = -1;
					newAReal(matLoc+d-1+halfP, matLoc+d-1+halfP) = -1;
					newAReal(finalMatLoc+d-1, finalMatLoc+d-1) = 1;
					newAReal(finalMatLoc+halfP+d-1, finalMatLoc+halfP+d-1) = 1;
				}
				
				// Subtract to force sum to identity
				newAReal(finalMatLoc+nextX, finalMatLoc+nextY) = -1;
				newAReal(finalMatLoc+nextY, finalMatLoc+nextX) = -1;
				newAReal(finalMatLoc+nextX+halfP, finalMatLoc+nextY+halfP) = -1;
				newAReal(finalMatLoc+nextY+halfP, finalMatLoc+nextX+halfP) = -1;

				// Move the location along
				nextX++;
				if (nextX >= d) {
					nextY++;
					nextX = nextY;
				}

				// Add this matrix to the list
				As.push_back(newAReal);

			}

			// For each imag vector element
			nextX = 1;
			nextY = 0;
			for (int l=0; l<numImagPer; l++) {

				// Create a blank p by p matrix
				Eigen::MatrixXd newAImag = Eigen::MatrixXd::Zero(p, p);

				// Place the imag comps in the off-diagonal blocks
				newAImag(matLoc+nextX+halfP, matLoc+nextY) = 1;
				newAImag(matLoc+nextY, matLoc+nextX+halfP) = 1;
				newAImag(matLoc+nextX, matLoc+nextY+halfP) = -1;
				newAImag(matLoc+nextY+halfP, matLoc+nextX) = -1;

				// Subtract to force sum to identity
				newAImag(finalMatLoc+nextX+halfP, finalMatLoc+nextY) = -1;
				newAImag(finalMatLoc+nextY, finalMatLoc+nextX+halfP) = -1;
				newAImag(finalMatLoc+nextX, finalMatLoc+nextY+halfP) = 1;
				newAImag(finalMatLoc+nextY+halfP, finalMatLoc+nextX) = 1;

				// Move the location along
				nextX++;
				if (nextX >= d) {
					nextY++;
					nextX = nextY+1;
				}

				// Add this matrix to the list
				As.push_back(newAImag);

			}

		}

	}

	// Calculate the B matrix
	B = Eigen::MatrixXd::Zero(p, p);
	for (int i=0; i<numMeasureB; i++) {
		for (int k=0; k<numOutcomeB-1; k++) {

			// The trace of each should be 1
			int matLoc = (i*numOutcomeB + k) * d;
			B(matLoc+d-1, matLoc+d-1) = 1;
			B(matLoc+halfP+d-1, matLoc+halfP+d-1) = 1;

		}

		// The last of each measurement should be the identity
		int matLoc = (i*numOutcomeB + numOutcomeB-1) * d;
		for (int a=0; a<d-1; a++) {
			B(matLoc+a, matLoc+a) = 1;
			B(matLoc+halfP+a, matLoc+halfP+a) = 1;
		}

	}

	// The interior point to optimise
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
	Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(p, p);

	// Initialise x TODO
	double ran = 0.30;
	//x << 1.0, 0.0,   0.0,     ran, std::sqrt(ran)*std::sqrt(1-ran),     0.0;

	// First measurement can just be ones
	for (int j=0; j<numOutcomeB-1; j++) {
		x(numTotalPer*j + j) = 1;
	}

	Eigen::MatrixXd XFinal2 = X(x, 0);
	precision = 1;
	prettyPrint("x = ", x);
	std::cout << std::endl;
	std::vector<Eigen::MatrixXcd> Ms2(numMeasureB*numOutcomeB, Eigen::MatrixXd::Zero(d,d));
	for (int i=0; i<numMeasureB; i++) {
		for (int j=0; j<numOutcomeB; j++) {
			int ind = i*numOutcomeB + j;
			Ms2[ind] = XFinal2.block(ind*d, ind*d, d, d) + 1i*XFinal2.block(ind*d+halfP, ind*d, d, d);
			std::cout << std::endl;
			prettyPrint("M_" + std::to_string(j) + "^" + std::to_string(i) + " = ", Ms2[ind]);
		}
	}
	return 0;

	// Initialise Z
	Eigen::MatrixXd XCached = X(x);
	Z = Eigen::MatrixXd::Zero(p, p);
	for (int i=0; i<numMeasureB; i++) {
		for (int j=0; j<numOutcomeB; j++) {
			int currentLoc = (i*numOutcomeB + j) * d;
			int copyLoc = (i*numOutcomeB + ((j+1) % numOutcomeB)) * d;
			Z.block(currentLoc,currentLoc,d,d) = XCached.block(copyLoc,copyLoc,d,d);
			Z.block(currentLoc+halfP,currentLoc+halfP,d,d) = XCached.block(copyLoc,copyLoc,d,d);
		}
	}

	// Cache things
	Eigen::MatrixXd XInverse = XCached.inverse();
	Eigen::MatrixXd ZInverse = Z.inverse();
	Eigen::MatrixXd gCached = g(x);
	Eigen::MatrixXd A_0 = delg(x);
	Eigen::VectorXd delfCached = delf(x);
	Eigen::VectorXd delLCached = delL(y, Z, delfCached, A_0);

	// To prevent reinit each time
	Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n,n);
	Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n, n);
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd AStarXInverse = Eigen::VectorXd::Zero(n);
	Eigen::MatrixXd GHInverse = Eigen::MatrixXd::Zero(n, n);
	Eigen::MatrixXd deltaX = Eigen::MatrixXd::Zero(n, n);
	Eigen::VectorXd deltax = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd deltay = Eigen::VectorXd::Zero(m);
	Eigen::MatrixXd deltaZ = Eigen::MatrixXd::Zero(p, p);

	// Outer loop
	double rMagZero = 0;
	double rMagMu = 0;
	int k = 0;
	int totalInner = 0;
	for (k=0; k<maxOuterIter; k++) {

		// Check if global convergence is reached
		rMagZero = rMag(0, Z, XCached, delLCached, gCached);
		if (rMagZero <= epsilon) {
			break;
		}

		// Outer-iteration output
		std::cout << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "       Iteration " << k << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "      -f(x) = " << -f(x) << " <= " << maxVal << std::endl;;
		std::cout << "      -L(x) = " << -L(x, y, Z) << std::endl;;
		std::cout << "         mu = " << mu << std::endl;;
		std::cout << "     r(w,0) = " << rMagZero << " ?< " << epsilon << std::endl;;
		std::cout << "--------------------------------" << std::endl;

		// Otherwise find the optimum for the current mu
		double epsilonPrime = M_c * mu;
		for (int k2=0; k2<maxInnerIter; k2++) {
			totalInner++;
		
			// Update the caches
			XCached = X(x);
			XInverse = XCached.inverse();
			ZInverse = Z.inverse();
			gCached = g(x);
			A_0 = delg(x);
			delfCached = delf(x);
			delLCached = delL(y, Z, delfCached, A_0);

			// Update T, the scaling matrix
			T = XCached.pow(-0.5);

			// Update G
			G = del2L(x, y);

			// See if G is already PD
			if (isPD(G)) {

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
			for (int i=0; i<n; i++) {
				for (int j=0; j<n; j++) {
					H(i,j) = (As[i]*XInverse*As[j]*Z).trace();
				}
			}

			// Calculate a few useful properties
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

			// Calculate optimal step size using a line search
			double alphaBarX = -gammaVal / std::real((XInverse * deltaX).eigenvalues()[0]);
			double alphaBarZ = -gammaVal / std::real((ZInverse * deltaZ).eigenvalues()[0]);
			if (alphaBarX < 0) {
				alphaBarX = 1;
			}
			if (alphaBarZ < 0) {
				alphaBarZ = 1;
			}
			double alphaBar = std::min(std::min(alphaBarX, alphaBarZ), 1.0);
			double alpha;
			for (int l=0; l<100; l++){
				alpha = alphaBar * std::pow(beta, l);
				if (F(x+alpha*deltax, Z+alpha*deltaZ, mu) <= F(x, Z, mu) + epsilonZero*alpha*deltaF(x, deltax, Z, deltaZ, mu) && isPD(X(x+alpha*deltax))) {
					break;
				}
			}

			// Update variables
			x += alpha*deltax;
			y += deltay;
			Z += alpha*deltaZ;
			
			// Inner-iteration output
			rMagMu = rMag(mu, Z, XCached, delLCached, gCached);
			std::cout << "f(x) = " << f(x) << " r = " << rMagMu  << " ?< " << epsilonPrime << " g(x) = " << gCached << std::endl;
			
			// Check if local convergence is reached
			if (rMagMu <= epsilonPrime) {
				break;
			}

		}

		// Update mu
		mu = mu / muScaling;

	}

	// Stop the timer
	auto t2 = std::chrono::high_resolution_clock::now();

	// Extract the solution from X
	std::cout << "" << std::endl;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "      Final Matrices " << std::endl;;
	std::cout << "--------------------------------" << std::endl;
	Eigen::MatrixXd XFinal = X(x, 0);
	std::vector<Eigen::MatrixXcd> Ms(numMeasureB*numOutcomeB, Eigen::MatrixXd::Zero(d,d));
	for (int i=0; i<numMeasureB; i++) {
		for (int j=0; j<numOutcomeB; j++) {
			int ind = i*numOutcomeB + j;
			Ms[ind] = XFinal.block(ind*d, ind*d, d, d) + 1i*XFinal.block(ind*d+halfP, ind*d, d, d);
			std::cout << std::endl;
			prettyPrint("M_" + std::to_string(j) + "^" + std::to_string(i) + " = ", Ms[ind]);
		}
	}

	// Final output
	std::cout << "" << std::endl;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "      Final Output " << std::endl;;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "    final r(w) = " << rMagZero << " < " << epsilon << std::endl;;
	std::cout << "    final f(x) = " << -f(x) << " <= " << maxVal << std::endl;;
	std::cout << "    final L(w) = " << -L(x, y, Z) << std::endl;;
	std::cout << "    final g(x) = " << gCached << std::endl;;
	std::cout << "   outer iters = " << k << std::endl;;
	std::cout << "   total inner = " << totalInner << std::endl;;
	std::cout << "          time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;

	// Everything went fine
	return 0;

}

