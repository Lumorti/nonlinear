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

// Optimisation parameters
double extraDiag = 0.001;
int maxOuterIter = 100000;
int maxInnerIter = 100000;
double muScaling = 2;

// Parameters between 0 and 1
double gammaVal = 0.9;
double epsilonZero = 0.9;
double beta = 0.9;

// Parameters greater than 0
double epsilon = 1.0;
double M_c = 0.1;
double mu = 1.0;
double nu = 0.1;

// An interior point has three components
class interiorPoint {
	public:
	Eigen::VectorXd x;
	Eigen::VectorXd y;
	Eigen::MatrixXd Z;
	interiorPoint(int n, int m, int p) {
		x = Eigen::VectorXd::Zero(n);
		y = Eigen::VectorXd::Zero(m);
		Z = Eigen::MatrixXd::Zero(p, p);
	}
};

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
// Pretty print a complex 2D dense Eigen array
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
			std::cout << std::setw(5) << arr(y,x);
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
			for (int k=0; k<numOutcomeB+1; k++) {
				for (int l=0; l<numOutcomeB+1; l++) {

					// Start locations of the real and imag submatrices
					int r1 = (i * (numOutcomeB+1) + k) * d;
					int r2 = (j * (numOutcomeB+1) + l) * d;

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

	// Extra term enforcing projectors TODO
	val -= XCached.cwiseProduct(XCached).sum() - p / 2;

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
						int r1 = (i * (numOutcomeB+1) + k) * d;
						int r2 = (j * (numOutcomeB+1) + l) * d;

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

		// Extra term enforcing projectors TODO
		vals(b) -= 2*As[b].cwiseProduct(XCached).sum();

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
							int r1 = (i * (numOutcomeB+1) + k) * d;
							int r2 = (j * (numOutcomeB+1) + l) * d;

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

			// Extra term enforcing projectors TODO
			vals(b,c) -= 2*As[b].cwiseProduct(As[c]).sum();

		}
	}

	// Return the matrix
	return vals;

}

// Constraint function
Eigen::VectorXd g(Eigen::VectorXd x) {

	// Vector to create
	Eigen::VectorXd gOutput = Eigen::VectorXd::Zero(m);

	// Return this vector of things that should be zero
	return gOutput;

}

// Gradient of the constraint function
Eigen::MatrixXd delg(Eigen::VectorXd x) {

	// Matrix representing the Jacobian of g
	Eigen::MatrixXd gOutput = Eigen::MatrixXd::Zero(m, n);

	// Return the gradient vector of g
	return gOutput;

}

// The Lagrangian 
double L(Eigen::VectorXd x, Eigen::VectorXd y, Eigen::MatrixXd Z) {
	return f(x) - y.transpose()*g(x) - X(x).cwiseProduct(Z).sum();
}

// Differential of the Lagrangian given individual components
Eigen::VectorXd delL(Eigen::VectorXd x, Eigen::VectorXd y, Eigen::MatrixXd Z) {

	// Start with a vector of n elements
	Eigen::VectorXd vals = Eigen::VectorXd::Zero(n);

	// Get the gradient of f(x)
	Eigen::VectorXd delfCached = delf(x);

	// Get A_0
	Eigen::MatrixXd A_0 = delg(x);

	// Calculate the first part as in the paper
	vals = delfCached - A_0.transpose() * y;

	// Then the second 
	for (int i=0; i<n; i++) {
		vals(i) -= As[i].cwiseProduct(Z).sum();
	}

	// Return this vector
	return vals;

}

// Double differential of the Lagrangian given an interior point
Eigen::MatrixXd del2L(interiorPoint w) {

	// In our case it's only the f(x) that has a non-zero value
	return del2f(w.x);

}

// Function giving the norm of a point, modified by some mu
double rMag(interiorPoint w, double mu) {

	// Calculate various vectors
	Eigen::MatrixXd XCached = X(w.x);
	Eigen::VectorXd delLCached = delL(w.x, w.y, w.Z);

	// The right part of the square root
	Eigen::MatrixXd XZI = XCached * w.Z - mu * Eigen::MatrixXd::Identity(p,p);

	// Sum the l^2/Frobenius norms
	double val = std::sqrt(delLCached.squaredNorm() + XZI.squaredNorm());

	// Return this magnitude
	return val;

}

// Handy functions for comparing the dimensions of two matrices
void dimCompare(Eigen::MatrixXd a, Eigen::MatrixXd b) {
	std::cout << "left dims = " << a.rows() << " x " << a.cols() << "   right dims = " << b.rows() << " x " << b.cols() << std::endl;
}

// The merit function
double F(Eigen::VectorXd x, Eigen::MatrixXd Z, double mu) {

	// Cache the X matrix
	Eigen::MatrixXd XCached = X(x);

	// Calculate the two components
	double FBP = f(x) - mu*log(XCached.determinant());
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

	// Calculate the two components
	double FBP = delf(x).transpose() * deltax - mu*(XInverse*deltaX).trace();
	double FPD = (deltaX*Z + XCached*deltaZ - mu*ZInverse*deltaZ - mu*XInverse*deltaX).trace();

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
	numOutcomeB = d-1;
	numUniquePer = (d*(d+1))/2-1;
	numRealPer = numUniquePer;
	numImagPer = numUniquePer+1-d;
	numTotalPer = numRealPer + numImagPer;

	// Sizes of matrices
	n = numMeasureB*numOutcomeB*numTotalPer;
	m = 0;
	p = numMeasureB*(numOutcomeB+1)*d*2;

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
		for (int k=0; k<numOutcomeB; k++) {

			// Where in X/x this matrix starts
			int matLoc = (i*(numOutcomeB+1) + k) * d;
			int finalMatLoc = (i*(numOutcomeB+1) + numOutcomeB) * d;
			int vecLocReal = (i*numOutcomeB + k) * numTotalPer;
			int vecLocImag = vecLocReal + numRealPer;
			int nextX = 0;
			int nextY = 0;

			// For each real vector element
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
		for (int k=0; k<numOutcomeB; k++) {

			// The trace of each should be 1
			int matLoc = (i*(numOutcomeB+1) + k) * d;
			B(matLoc+d-1, matLoc+d-1) = 1;
			B(matLoc+halfP+d-1, matLoc+halfP+d-1) = 1;

		}

		// The last of each measurement should be the identity
		int matLoc = (i*(numOutcomeB+1) + numOutcomeB) * d;
		for (int a=0; a<d-1; a++) {
			B(matLoc+a, matLoc+a) = 1;
			B(matLoc+halfP+a, matLoc+halfP+a) = 1;
		}

	}

	// The interior point to optimise
	interiorPoint w(n, m, p);

	// Initialise x
	srand((unsigned int) time(0));
	//w.x << 1.0, 0.0,   0.0,     0.6, 0.5,     0.0;
	w.x << 1.0, 0.0,   0.0,     0.5, 0.5,     0.0;
	//w.x = Eigen::VectorXd::Random(n);

	// Initialise Z
	w.Z = Eigen::MatrixXd::Identity(p, p);

	// G, the Hessian of L(w)
	Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n,n);

	// Outer loop
	double rMagZero = 0;
	double rMagMu = 0;
	int k = 0;
	for (k=0; k<maxOuterIter; k++) {

		// Check if global convergence is reached
		rMagZero = rMag(w, 0);
		if (rMagZero <= epsilon) {
			break;
		}

		// Outer-iteration output
		std::cout << "" << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "       Iteration " << k << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "       f(x) = " << -f(w.x) << " <= " << maxVal << std::endl;;
		std::cout << "       L(x) = " << -L(w.x, w.y, w.Z) << std::endl;;
		std::cout << "         mu = " << mu << std::endl;;
		std::cout << "  rMag(w,0) = " << rMagZero << " ?< " << epsilon << std::endl;;
		std::cout << "--------------------------------" << std::endl;

		// Otherwise find the optimum for the current mu
		double epsilonPrime = M_c * mu;
		for (int k2=0; k2<maxInnerIter; k2++) {
		
			// Cache things
			Eigen::MatrixXd XCached = X(w.x);
			Eigen::MatrixXd XInverse = XCached.inverse();
			Eigen::MatrixXd ZInverse = w.Z.inverse();
			Eigen::MatrixXd A_0 = delg(w.x);
			Eigen::VectorXd delLCached = delL(w.x, w.y, w.Z);
			Eigen::VectorXd delfCached = delf(w.x);

			// Calculate T, the scaling matrix
			Eigen::MatrixXd T = XCached.pow(-0.5);

			// Update G
			G = del2L(w);

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
			Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
			for (int i=0; i<n; i++) {
				for (int j=0; j<n; j++) {
					H(i,j) = (As[i]*XInverse*As[j]*w.Z).trace();
				}
			}

			// Calculate direction
			interiorPoint delta(n, m, p);
			Eigen::MatrixXd GHInverse = (G + H).inverse();
			Eigen::VectorXd AStarXInverse = Eigen::VectorXd::Zero(n);
			for (int i=0; i<n; i++) {
				AStarXInverse(i) = As[i].cwiseProduct(XInverse).sum();
			}
			delta.x = GHInverse * (-delfCached + mu*AStarXInverse);
			Eigen::MatrixXd deltaX = XNoB(delta.x);
			delta.Z = mu*XInverse - w.Z - 0.5*(XInverse*deltaX*w.Z + w.Z*deltaX*XInverse);

			// Calculate optimal step size using a line search
			double alphaBarX = 1.0;
			double alphaBarZ = 1.0;
			alphaBarX = -gammaVal / std::real((XInverse * deltaX).eigenvalues()[0]);
			alphaBarZ = -gammaVal / std::real((ZInverse * delta.Z).eigenvalues()[0]);
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
				if (F(w.x+alpha*delta.x, w.Z+alpha*delta.Z, mu) <= F(w.x, w.Z, mu) + epsilonZero*alpha*deltaF(w.x, delta.x, w.Z, delta.Z, mu) && isPD(X(w.x+alpha*delta.x))) {
					break;
				}
			}

			// Update variables
			w.x += alpha*delta.x;
			w.y += delta.y;
			w.Z += alpha*delta.Z;
			
			// Inner-iteration output
			std::cout << " f(x) = " << -f(w.x) << " r = " << rMagMu  << " ?< " << epsilonPrime << " L(w) = " << L(w.x, w.y, w.Z) << std::endl;
			
			// Check if local convergence is reached
			rMagMu = rMag(w, mu);
			if (rMagMu <= M_c*mu) {
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
	Eigen::MatrixXd XFinal = X(w.x, 0);
	std::vector<Eigen::MatrixXcd> Ms(numMeasureB*(numOutcomeB+1), Eigen::MatrixXd::Zero(d,d));
	for (int i=0; i<numMeasureB; i++) {
		for (int j=0; j<numOutcomeB+1; j++) {
			int ind = i*(numOutcomeB+1) + j;
			Ms[ind] = XFinal.block(ind*d, ind*d, d, d) + 1i*XFinal.block(ind*d+halfP, ind*d, d, d);
			prettyPrint("M_" + std::to_string(j) + "^" + std::to_string(i) + " = ", Ms[ind]);
		}
	}

	// Final output
	std::cout << "" << std::endl;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "      Final Output " << std::endl;;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "    final r(w) = " << rMag(w, 0) << " < " << epsilon << std::endl;;
	std::cout << "    final f(x) = " << -f(w.x) << " <= " << maxVal << std::endl;;
	std::cout << "    final L(w) = " << -L(w.x, w.y, w.Z) << std::endl;;
	std::cout << "    iterations = " << k << std::endl;;
	std::cout << "          time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;

	// Everything went fine
	return 0;

}

