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
double fScaling = 1;
double gScaling = 1;
double gThresh = 1e-10;
int numCores = 1;
bool useBFGS = true;
double BFGSmaxG = 0.1;
bool experimental = false;

// Parameters between 0 and 1
double gammaVal = 0.9;
double epsilonZero = 0.9;
double beta = 0.1;
double nu = 0.9;
double rho = 0.5;

// Parameters greater than 0
double epsilon = 1e-5; 
double M_c = 1e10;
double mu = 1.0;

// Useful global quantities
int numPerm = 0;
int numMeasureB = 0;
int numOutcomeB = 0;
int numUniquePer = 0;
int numRealPer = 0;
int numImagPer = 0;
int numTotalPer = 0;
std::vector<Eigen::SparseMatrix<double>> As;
Eigen::MatrixXi startEndA;
Eigen::SparseMatrix<double> B;

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

	// Add a bit extra to make it invertible
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

// Double differential of the objective function
Eigen::MatrixXd del2fOld(Eigen::SparseMatrix<double> XZero) {

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

// Double differential of the objective function
Eigen::MatrixXd del2f(Eigen::SparseMatrix<double> XZero) {

	// Create an n by n matrix of all zeros
	Eigen::MatrixXd vals = Eigen::MatrixXd::Zero(n, n);

	#pragma omp parallel
	{

	// Init some things here for speed
	int ind1 = 0;
	int ind2 = 0;
	int r1 = 0;
	int r2 = 0;
	double prodReal = 0;
	double prodImag = 0;
	double den = 0;
	double coeff1 = 0;
	double coeff2 = 0;
	double BXsum = 0;
	double CXsum = 0;
	double BCsum = 0;

	// For each pair of measurements
	#pragma omp for
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
	
			// For each outcome of these measurements
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// Start locations of the real and imag submatrices
					ind1 = i*numOutcomeB + k;
					ind2 = j*numOutcomeB + l;
					r1 = ind1*d;
					r2 = ind2*d;

					// Extract the blocks for this i, j
					Eigen::SparseMatrix<double> XReal1 = XZero.block(r1,r1,d,d);
					Eigen::SparseMatrix<double> XReal2 = XZero.block(r2,r2,d,d);
					Eigen::SparseMatrix<double> XImag1 = XZero.block(r1+halfP,r1,d,d);
					Eigen::SparseMatrix<double> XImag2 = XZero.block(r2,r2+halfP,d,d);

					// The original value inside the square root
					prodReal = XReal1.cwiseProduct(XReal2).sum();
					prodImag = XImag1.cwiseProduct(XImag2).sum();
					den = 1.0 - prodReal + prodImag;
					coeff1 = 0.25 * std::pow(den, -1.5);
					coeff2 = 0.5 * std::pow(den, -0.5);

					// For each component of the vector
					for (int b=startEndA(ind1, 0); b<startEndA(ind1, 1); b++) {

						// Extract the blocks for this b
						Eigen::SparseMatrix<double> BReal1 = As[b].block(r1,r1,d,d);
						Eigen::SparseMatrix<double> BReal2 = As[b].block(r2,r2,d,d);
						Eigen::SparseMatrix<double> BImag1 = As[b].block(r1+halfP,r1,d,d);
						Eigen::SparseMatrix<double> BImag2 = As[b].block(r2,r2+halfP,d,d);

						// Components with As[b] and X
						BXsum = BImag1.cwiseProduct(XImag2).sum() + XImag1.cwiseProduct(BImag2).sum() - BReal1.cwiseProduct(XReal2).sum() - XReal1.cwiseProduct(BReal2).sum();

						// For each component of the vector
						for (int c=startEndA(ind1, 0); c<startEndA(ind1, 1); c++) {

							// Extract the blocks for this c
							Eigen::SparseMatrix<double> CReal1 = As[c].block(r1,r1,d,d);
							Eigen::SparseMatrix<double> CReal2 = As[c].block(r2,r2,d,d);
							Eigen::SparseMatrix<double> CImag1 = As[c].block(r1+halfP,r1,d,d);
							Eigen::SparseMatrix<double> CImag2 = As[c].block(r2,r2+halfP,d,d);

							// Components with As[c] and X
							CXsum = CImag1.cwiseProduct(XImag2).sum() + XImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(XReal2).sum() - XReal1.cwiseProduct(CReal2).sum();

							// Components with As[b] and As[c]
							BCsum = CImag1.cwiseProduct(BImag2).sum() + BImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(BReal2).sum() - BReal1.cwiseProduct(CReal2).sum();

							// Add this inner product of the submatrices
							vals(b, c) += coeff1*BXsum*CXsum - coeff2*BCsum;

						}

						// For each component of the vector
						for (int c=startEndA(ind2, 0); c<startEndA(ind2, 1); c++) {

							// Extract the blocks for this c
							Eigen::SparseMatrix<double> CReal1 = As[c].block(r1,r1,d,d);
							Eigen::SparseMatrix<double> CReal2 = As[c].block(r2,r2,d,d);
							Eigen::SparseMatrix<double> CImag1 = As[c].block(r1+halfP,r1,d,d);
							Eigen::SparseMatrix<double> CImag2 = As[c].block(r2,r2+halfP,d,d);

							// Components with As[c] and X
							CXsum = CImag1.cwiseProduct(XImag2).sum() + XImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(XReal2).sum() - XReal1.cwiseProduct(CReal2).sum();

							// Components with As[b] and As[c]
							BCsum = CImag1.cwiseProduct(BImag2).sum() + BImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(BReal2).sum() - BReal1.cwiseProduct(CReal2).sum();

							// Add this inner product of the submatrices
							vals(b, c) += coeff1*BXsum*CXsum - coeff2*BCsum;

						}

					}

					// For each component of the vector
					for (int b=startEndA(ind2, 0); b<startEndA(ind2, 1); b++) {

						// Extract the blocks for this b
						Eigen::SparseMatrix<double> BReal1 = As[b].block(r1,r1,d,d);
						Eigen::SparseMatrix<double> BReal2 = As[b].block(r2,r2,d,d);
						Eigen::SparseMatrix<double> BImag1 = As[b].block(r1+halfP,r1,d,d);
						Eigen::SparseMatrix<double> BImag2 = As[b].block(r2,r2+halfP,d,d);

						// Components with As[b] and X
						BXsum = BImag1.cwiseProduct(XImag2).sum() + XImag1.cwiseProduct(BImag2).sum() - BReal1.cwiseProduct(XReal2).sum() - XReal1.cwiseProduct(BReal2).sum();

						// For each component of the vector
						for (int c=startEndA(ind1, 0); c<startEndA(ind1, 1); c++) {

							// Extract the blocks for this c
							Eigen::SparseMatrix<double> CReal1 = As[c].block(r1,r1,d,d);
							Eigen::SparseMatrix<double> CReal2 = As[c].block(r2,r2,d,d);
							Eigen::SparseMatrix<double> CImag1 = As[c].block(r1+halfP,r1,d,d);
							Eigen::SparseMatrix<double> CImag2 = As[c].block(r2,r2+halfP,d,d);

							// Components with As[c] and X
							CXsum = CImag1.cwiseProduct(XImag2).sum() + XImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(XReal2).sum() - XReal1.cwiseProduct(CReal2).sum();

							// Components with As[b] and As[c]
							BCsum = CImag1.cwiseProduct(BImag2).sum() + BImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(BReal2).sum() - BReal1.cwiseProduct(CReal2).sum();

							// Add this inner product of the submatrices
							vals(b, c) += coeff1*BXsum*CXsum - coeff2*BCsum;

						}

						// For each component of the vector
						for (int c=startEndA(ind2, 0); c<startEndA(ind2, 1); c++) {

							// Extract the blocks for this c
							Eigen::SparseMatrix<double> CReal1 = As[c].block(r1,r1,d,d);
							Eigen::SparseMatrix<double> CReal2 = As[c].block(r2,r2,d,d);
							Eigen::SparseMatrix<double> CImag1 = As[c].block(r1+halfP,r1,d,d);
							Eigen::SparseMatrix<double> CImag2 = As[c].block(r2,r2+halfP,d,d);

							// Components with As[c] and X
							CXsum = CImag1.cwiseProduct(XImag2).sum() + XImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(XReal2).sum() - XReal1.cwiseProduct(CReal2).sum();

							// Components with As[b] and As[c]
							BCsum = CImag1.cwiseProduct(BImag2).sum() + BImag1.cwiseProduct(CImag2).sum() - CReal1.cwiseProduct(BReal2).sum() - BReal1.cwiseProduct(CReal2).sum();

							// Add this inner product of the submatrices
							vals(b, c) += coeff1*BXsum*CXsum - coeff2*BCsum;

						}

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
    #pragma omp parallel for
	for (int i=0; i<n; i++) {
		gOutput(0, i) = 2*GCached.cwiseProduct(2*As[i]*XZero - As[i]).sum();
	}

	// Return the gradient vector of g
	return gScaling*gOutput;

}

// Second derivatives of the constraint function if m=1
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
Eigen::VectorXd delL(Eigen::VectorXd y, Eigen::SparseMatrix<double> ZSparse, Eigen::VectorXd delfCached, Eigen::MatrixXd A_0) {

	// Calculate A* Z
	Eigen::VectorXd AStarZ = Eigen::VectorXd::Zero(n);
	for (int i=0; i<n; i++) {
		AStarZ(i) = As[i].cwiseProduct(ZSparse).sum();
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
double rMag(double mu, Eigen::SparseMatrix<double> ZSparse, Eigen::SparseMatrix<double> XCached, Eigen::VectorXd delLCached, Eigen::VectorXd gCached) {

	// The left part of the square root
	Eigen::VectorXd left = Eigen::VectorXd::Zero(n+m);
	left << delLCached, gCached;

	// The right part of the square root
	Eigen::MatrixXd right = XCached * ZSparse - mu * Eigen::MatrixXd::Identity(p,p);

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
	return G.llt().info() != Eigen::NumericalIssue;
}
bool isComplexPD(Eigen::MatrixXcd G) {
	return G.llt().info() != Eigen::NumericalIssue;
}

// Given a matrix, make it be positive definite
void makePD(Eigen::MatrixXd G) {

	// See if G is already PD
	if (!isPD(G)) {

		// If G+sigma*I is PD
		double sigma = 1;
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
		if (isPD(G+sigma*I)) {

			// Decrease sigma until it isn't
			while (isPD(G+sigma*I) && sigma >= 1e-7) {
				sigma /= 2;
			}

			// Then return to the one that was still PD
			sigma *= 2;

		// If G+sigma*I is not PD
		} else {

			// Increase sigma until it is
			while (!isPD(G+sigma*I)) {
				sigma *= 2;
			}

		}

		// Update this new G
		G = G+sigma*I;

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
			std::cout << " -x               use experimental features" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "       output options          " << std::endl;
			std::cout << " -p [int]         set the precision" << std::endl;
			std::cout << " -B               output only the iter count" << std::endl;
			std::cout << " -K               output only the time taken" << std::endl;
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

		// If told to use experimental features
		} else if (arg == "-x") {
			experimental = true;

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

		// If told to only output the iteration count
		} else if (arg == "-K") {
			outputMode = "K";

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
	numUniquePer = (d*(d+1))/2-1;
	numRealPer = numUniquePer;
	numImagPer = numUniquePer+1-d;
	numTotalPer = numRealPer + numImagPer;

	// Sizes of matrices
	n = numMeasureB*(numOutcomeB-1)*numTotalPer;
	m = 1;
	p = numMeasureB*numOutcomeB*d*2;
	halfP = p / 2;

	// The "ideal" value
	double maxVal = fScaling*numPerm*d*std::sqrt(d*(d-1));

	// Output various bits of info about the problem/parameters
	if (outputMode == "") {
		std::cout << "--------------------------------" << std::endl;
		std::cout << "          System Info           " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "               d = " << d << std::endl;
		std::cout << "               n = " << sets << std::endl;
		std::cout << "  size of vector = " << n << std::endl;
		std::cout << "  size of matrix = " << p << " x " << p << " ~ " << p*p*16 / (1024*1024) << " MB " << std::endl;
		std::cout << "     ideal value = " << maxVal << std::endl;
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
		std::cout << "           cores = " << numCores << std::endl;
		std::cout << "" << std::endl;
	}

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

	// Cache whether each A matrix has non zero elements at a certain matrix index
	startEndA = -Eigen::MatrixXi::Ones(numMeasureB*numOutcomeB, 2);
	for (int i=0; i<numMeasureB; i++) {
		for (int k=0; k<numOutcomeB; k++) {
			for (int b=0; b<n; b++) {
				int ind1 = i*numOutcomeB + k;
				int r1 = ind1*d;
				Eigen::SparseMatrix<double> BReal1 = As[b].block(r1,r1,d,d);
				Eigen::SparseMatrix<double> BImag1 = As[b].block(r1+halfP,r1,d,d);
				if (BReal1.nonZeros() + BImag1.nonZeros() != 0) {
					if (startEndA(ind1, 0) == -1) {
						startEndA(ind1, 0) = b;
					}
					startEndA(ind1, 1) = b+1;
				}
			}
		}
	}

	// The interior point to optimise
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
	Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(p, p);

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
				XDense.block(ind+halfP, ind+halfP, d, d) = tempMat.real();
				XDense.block(ind, ind+halfP, d, d) = tempMat.imag();
				XDense.block(ind+halfP, ind, d, d) = tempMat.imag();

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

		// Turn these into an X
		for (int i=0; i<Ms.size(); i++) {
			int ind = i*d;
			for (int j=0; j<d; j++) {
				for (int k=0; k<d; k++) {
					XDense(ind+j, ind+k) = std::real(Ms[i][j][k]);
					XDense(ind+halfP+j, ind+halfP+k) = std::real(Ms[i][j][k]);
					XDense(ind+j+halfP, ind+k) = -std::imag(Ms[i][j][k]);
					XDense(ind+j, ind+halfP+k) = std::imag(Ms[i][j][k]);
				}
			}
		}

		// Then this X into an x
		x = Xtox(XDense);

	} 
	
	// View part of the search space
	if (outputMode == "vis") {

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

		// Make it an interior point
		double v = 0;
		if (gScaling == -1) {
			gScaling = 0.1;
		}
		for (int i=0; i<10000000; i++) {
			XZero = X(x, 0.0);
			v = std::abs(g(XZero)(0) / gScaling);
			if (v < gThresh) {
				break;
			}
			x -= delg(XZero).row(0);
		}

		// Save this og val
		Eigen::VectorXd oldx = x;

		// Settings for the vis
		double minDel = -1.0;
		double maxDel = 1.0;
		double stepsPer = 20;
		double delPer = (maxDel-minDel) / stepsPer;
		Eigen::MatrixXd delVec = Eigen::MatrixXd::Zero(m, n);

		// Look either side of the point
		for (double del1=minDel; del1<maxDel; del1+=delPer) {
			for (double del2=minDel; del2<maxDel; del2+=delPer) {

				// Adjust it from the og
				x(x1) = oldx(x1) + del1;
				x(x2) = oldx(x2) + del2;

				// Make it an interior point
				double v = 0;
				for (int i=0; i<1000; i++) {
					XZero = X(x, 0.0);
					v = std::abs(g(XZero)(0) / gScaling);
					if (v < gThresh) {
						break;
					}
					delVec = delg(XZero);
					x(x1) -= delVec(0,x1);
					x(x2) -= delVec(0,x2);
				}

				// Output g(x) and f(x)
				std::cout << x(x1) << " " << x(x2) << " " << f(X(x, 0.0)) << std::endl;

			}
		}

		// Then stop
		return 0;

	}

	// Save the current x and gScaling values
	Eigen::VectorXd oldx = x;
	double gradScaling = 1;
	double origScaling = gScaling;

	// Keep trying descent until we find the largest step size that works
	for (int j=0; j=100; j++) {

		// Reset x
		x = oldx;
		gScaling = gradScaling;

		// Gradient descent to make sure we start with an interior point
		double v = 0;
		for (int i=0; i<10000000; i++) {
			XZero = X(x, 0.0);
			v = std::abs(g(XZero)(0) / gradScaling);
			if (outputMode == "") {
				std::cout << "g(x) = " << v << std::endl;
			}
			if (v < gThresh || std::isnan(v)) {
				break;
			}
			x -= delg(XZero).row(0);
		}

		// If it wasn't nan, all is good
		if (!std::isnan(v)) {
			break;
		}

		// Otherwise decrease the scaling
		if (outputMode == "") {
			std::cout << "decreasing g factor from " << gScaling << " to " << gScaling / 1.5 << std::endl;
		}
		gradScaling /= 1.2;

	}

	// If gScaling wasn't manually set, then use the gradScaling
	if (origScaling == -1) {
		gScaling = gradScaling;
	} else {
		gScaling = origScaling;
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
				std::cout << "tr(M^2-M)  = " << (M.adjoint()*M - M).trace() << std::endl;
				std::cout << "is M PD? = " << isComplexPD(M) << std::endl;
			}
		}
	}

	// Some info about the initial X
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "|X^2-X|  = " << (XZero*XZero - XZero).squaredNorm() << std::endl;
		std::cout << "tr(X^2-X)  = " << trace(XZero*XZero - XZero) << std::endl;
		std::cout << "isPD(X) = " << isPD(XZero) << std::endl;
	}

	// Ensure this is an interior point
	if (std::abs(g(XZero)(0) / gScaling) > gThresh) {
		std::cerr << "Error - X should start as an interior point (g(x) = " << std::abs(g(XZero)(0) / gScaling) << " > gThresh)" << std::endl;
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
	Eigen::SparseMatrix<double> ZSparse = Z.sparseView();
	Eigen::VectorXd gCached = g(XZero);
	Eigen::MatrixXd A_0 = delg(XZero);
	Eigen::VectorXd delfCached = delf(XZero);
	Eigen::VectorXd delLCached = delL(y, ZSparse, delfCached, A_0);
	double rMagZero = rMag(0, ZSparse, XCached, delLCached, gCached);

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
		G = del2L(XZero, y);

		// Ensure it's positive definite
		makePD(G);

	}

	// Outer loop
	double rMagMu = 0;
	int k = 0;
	int totalInner = 0;
	for (k=0; k<maxOuterIter; k++) {

		// Check if global convergence is reached
		rMagZero = rMag(0, ZSparse, XCached, delLCached, gCached);
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
			XCached = X(x);
			XZero = X(x, 0.0);
			XDense = Eigen::MatrixXd(XCached);
			ZSparse = Z.sparseView();
			XInverse = XDense.inverse().sparseView();
			ZInverse = Z.inverse().sparseView();
			gCached = g(XZero);
			A_0 = delg(XZero);
			delfCached = delf(XZero);
			delLCached = delL(y, ZSparse, delfCached, A_0);

			// If not doing BFGS, need to do a full re-calc of G
			if (!useBFGS || gCached(0)/gScaling > BFGSmaxG) {

				// Update G
				G = del2L(XZero, y);

				// Ensure it's positive definite
				makePD(G);

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
			if (experimental) {
				Eigen::BiCGSTAB<Eigen::MatrixXd> solver(leftMat);
				solution = solver.solve(rightVec);
			} else {
				solution = leftMat.colPivHouseholderQr().solve(rightVec); // TODO try different
			}
			deltax = solution.head(n);
			deltay = solution.tail(m);

			// Then calculate the Z
			deltaX = XNoB(deltax);
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
			double FCached = F(x, ZSparse, mu);
			double deltaFCached = deltaF(deltaZ, ZInverse, ZSparse, XCached, XInverse, delfCached, gCached, A_0, deltax);
			for (l=0; l<maxL; l++){
				alpha = alphaBar * std::pow(beta, l);
				if (F(x+alpha*deltax, ZSparse+alpha*deltaZ, mu) <= FCached + epsilonZero*alpha*deltaFCached && isPD(X(x+alpha*deltax))) {
					break;
				}
			}

			// Inner-iteration output
			rMagMu = rMag(mu, ZSparse, XCached, delLCached, gCached);
			if (outputMode == "") {

				// Output the line
				std::cout << "f = " << f(XZero) << "   r = " << rMagMu  << " ?< " << epsilonPrime << "   g = " << gCached(0)/gScaling << std::endl;

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
			Z += alpha*deltaZ;
			
			// If using a BFGS update
			if (useBFGS) {

				// Update certain quantities
				XZero = X(x, 0.0);
				A_0 = delg(XZero);
				delfCached = delf(XZero);
				ZSparse = Z.sparseView();

				// Update G
				Eigen::VectorXd s = x - prevx;
				Eigen::VectorXd q = delL(y, ZSparse, delfCached, A_0) - delL(y, ZSparse, prevDelfCached, prevA_0);
				double psi = 1;
				if (s.dot(q) < 0.2*s.dot(G*s)) {
					psi = (0.8*s.dot(G*s)) / (s.dot(G*s-q));
				}
				Eigen::VectorXd qBar = psi*q + (1-psi)*(G*s);
				G = G - ((G*(s*(s.transpose()*G))) / (s.dot(G*s))) + ((qBar*qBar.transpose()) / (s.dot(qBar)));

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
	if (outputMode == "") {
		std::cout << "" << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "      Final Z " << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		for (int i=0; i<numMeasureB; i++) {
			for (int j=0; j<numOutcomeB; j++) {
				int ind = (i*numOutcomeB + j)*d;
				Eigen::MatrixXcd M = Z.block(ind, ind, d, d) + 1i*Z.block(ind+halfP, ind, d, d);
				std::cout << std::endl;
				prettyPrint("Z_" + std::to_string(j) + "^" + std::to_string(i) + " = ", M);
			}
		}
	}

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
		std::cout << "        |r(w)| = " << rMagZero << " < " << epsilon << std::endl;;
		std::cout << "         -f(x) = " << -f(XZero)/fScaling << " <= " << maxVal << std::endl;;
		std::cout << "          g(x) = " << gCached(0)/gScaling << std::endl;;
		std::cout << "         -L(w) = " << -L(XZero, XCached, y, Z) << std::endl;;
		std::cout << "         <X,Z> = " << XCached.cwiseProduct(ZSparse).sum() << std::endl;;
		std::cout << "           |y| = " << y.norm() << std::endl;;
		std::cout << "      y^T*g(x) = " << y.transpose()*g(XZero) << std::endl;;
		std::cout << "     |delf(x)| = " << (delfCached/fScaling).norm() << std::endl;;
		std::cout << "     |delL(w)| = " << delLCached.norm() << std::endl;;
		std::cout << "     |delg(x)| = " << (A_0/gScaling).norm() << std::endl;;
		std::cout << "    |del2f(x)| = " << (del2f(XZero)/fScaling).norm() << std::endl;;
		std::cout << "    |del2L(w)| = " << G.norm() << std::endl;;
		std::cout << "    |del2g(x)| = " << (del2g(XZero)/gScaling).norm() << std::endl;;
		std::cout << "   total inner = " << totalInner << std::endl;;
		std::cout << "    time taken = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;

	// Only output the total iterations required
	} else if (outputMode == "B") {
		std::cout << totalInner << std::endl;

	// Only output the total time required
	} else if (outputMode == "K") {
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

	}

	// Everything went fine
	return 0;

}

