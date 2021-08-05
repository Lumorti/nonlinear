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
int precision = 3;

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

// Get the smallest eigenvalue of a matrix
double smallestEigenvalue(Eigen::MatrixXd A) {

	// Start with a random unit vector
	Eigen::VectorXd w = Eigen::VectorXd::Random(A.cols());
	Eigen::VectorXd v = w.normalized();

	// We want the lowest not the highest
	Eigen::MatrixXd AInv = A.inverse();

	// Iteratively solve Aw = v, 
	double eigenval = 1.0 / w.norm();
	for (int i=0; i<10; i++) {
		w = AInv * v;
		eigenval = 1.0 / w.norm();
		v = w.normalized();
	}

	// Return the minimum eigenvalue of A
	return eigenval;

}

// Function turning x to X
Eigen::MatrixXd X(Eigen::VectorXd x, double extra=0.01) {

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

					// For readability
					double prodReal = XCached.block(r1,r1,d,d).cwiseProduct(XCached.block(r2,r2,d,d)).sum();
					double prodImag = XCached.block(r1+halfP,r1,d,d).cwiseProduct(XCached.block(r2,r2+halfP,d,d)).sum();
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

// Gradient of the objective function TODO
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

	}

	// Return the function value
	return vals;

}

// Double differential of the objective function TODO
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

		}
	}

	// Return the matrix
	return vals;

}

// Gradient of the constraint function
Eigen::MatrixXd delG(Eigen::VectorXd x) {

	// Matrix representing the Jacobian of g
	Eigen::MatrixXd gOutput = Eigen::MatrixXd::Zero(m, n);

	// Return the gradient vector of g
	return gOutput;

}

// Differential of the Lagrangian given individual components
Eigen::VectorXd delL(Eigen::VectorXd x, Eigen::VectorXd y, Eigen::MatrixXd Z) {

	// Start with a vector of n elements
	Eigen::VectorXd vals = Eigen::VectorXd::Zero(n);

	// Get the gradient of f(x)
	Eigen::VectorXd delfCached = delf(x);

	// Get A_0
	Eigen::MatrixXd A_0 = delG(x);

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

// Constraint function
Eigen::VectorXd g(Eigen::VectorXd x) {

	// Vector to create
	Eigen::VectorXd gOutput = Eigen::VectorXd::Zero(m);

	// Return this vector of things that should be zero
	return gOutput;

}

// Function giving the norm of a point, modified by some mu
double rMag(interiorPoint w, double mu) {

	// Calculate various vectors
	Eigen::MatrixXd XCached = X(w.x);
	Eigen::VectorXd gCached = g(w.x);
	Eigen::VectorXd delLCached = delL(w.x, w.y, w.Z);

	// Combined g and delL for the left part of the square root
	Eigen::VectorXd combined(gCached.size() + delLCached.size());
	combined << delLCached, gCached;

	// The right part of the square root
	Eigen::MatrixXd XZI = XCached * w.Z - mu * Eigen::MatrixXd::Identity(p,p);

	// Sum the l^2/Frobenius norms
	double val = std::sqrt(combined.squaredNorm() + XZI.squaredNorm());

	// Return this magnitude
	return val;

}

// Handy functions for comparing the dimensions of two matrices
void dimCompare(Eigen::MatrixXd a, Eigen::MatrixXd b) {
	std::cout << "left dims = " << a.rows() << " x " << a.cols() << "   right dims = " << b.rows() << " x " << b.cols() << std::endl;
}

// Standard cpp entry point
int main(int argc, char ** argv) {

	// Start the timer 
	auto t1 = std::chrono::high_resolution_clock::now();

	// Defining the MUB problem
	d = 2;
	sets = 2;

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

	// Optimisation parameters
	double epsilon = 1e-5;
	double M_c = 0.1;
	double mu = 1.0;
	double gamma = 0.9;
	double beta = 0.1;

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
	w.x << 1.0, 0.0,   0.0,     0.6, 0.5,     0.0;
	//srand((unsigned int) time(0));
	//w.x = Eigen::VectorXd::Random(n);

	// Initialise Z
	w.Z = Eigen::MatrixXd::Identity(p, p);

	// Calculate the initial G, the Hessian of L(w)
	Eigen::MatrixXd G = del2L(w);

	// Test new X(x) TODO
	prettyPrint("X(x) = ", X(w.x));
	std::cout << "f(x) = " <<  f(w.x) << std::endl;
	prettyPrint("delf(x) = ", delf(w.x));
	prettyPrint("del2f(x) = ", del2f(w.x));

	// See if G isn't PSD
	if (G.ldlt().info() != Eigen::ComputationInfo::Success) {

		// If G-sigma*I is PSD
		double sigma = 1;
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
		if ((G-sigma*I).ldlt().info() == Eigen::ComputationInfo::Success) {

			// Decrease sigma until it isn't
			while ((G-sigma*I).ldlt().info() == Eigen::ComputationInfo::Success) {
				sigma /= 2;
			}

			// Then return to the one that was still PSD
			sigma *= 2;

		// If G-sigma*I is not PSD
	 	} else {

			// Increase sigma until it is
			while ((G-sigma*I).ldlt().info() != Eigen::ComputationInfo::Success) {
				sigma *= 2;
			}

		}

		// Update this new G
		G = G-sigma*I;

	}
	return 0;
			
	// Outer loop
	int maxIter = 3;
	double rMagZero = 0;
	double rMagMu = 0;
	int k = 0;
	double epsilonPrime = M_c * epsilon;
	Eigen::VectorXd prevx = w.x;
	for (k=0; k<maxIter; k++) {

		// Check if global convergence is reached
		rMagZero = rMag(w, 0);
		if (rMagZero <= epsilonPrime) {
			break;
		}

		// Otherwise find the optimum for the current mu
		for (int k2=0; k2<maxIter; k2++) {
		
			// Check if local convergence is reached
			rMagMu = rMag(w, mu);
			if (rMagMu <= epsilonPrime) {
				break;
			}

			// Cache things
			Eigen::MatrixXd XCached = X(w.x);
			Eigen::MatrixXd XInverse = XCached.inverse();
			Eigen::MatrixXd A_0 = delG(w.x);
			Eigen::VectorXd delLCached = delL(w.x, w.y, w.Z);

			// Calculate T, the scaling matrix
			Eigen::MatrixXd T = XCached.pow(-0.5);

			// Update G
			Eigen::VectorXd q = delLCached - delL(prevx, w.y, w.Z);
			Eigen::VectorXd s = w.x - prevx;
			Eigen::MatrixXd sTran = s.transpose();
			double psi = 1;
			if (s.dot(q) < 0.2*s.dot(G*s)) {
				psi = (0.8*s.dot(G*s)) / (s.dot(G*s-q));
			}
			Eigen::VectorXd qBar = psi*q + (1-psi)*(G*s);
			G += -((G*(s*(sTran*G))) / (s.dot(G*s))) + ((qBar*qBar.transpose()) / (s.dot(qBar)));
	std::cout << "here8" << std::endl;

			// Calculate direction TODO
			interiorPoint delta(n, m, p);
			delta.x = -g(w.x)*A_0.inverse();
	std::cout << "here8.1" << std::endl;
			Eigen::MatrixXd DeltaX = X(delta.x);
	std::cout << "here8.2" << std::endl;
			delta.Z = mu*XInverse - w.Z - (XInverse*(DeltaX*w.Z) + w.Z*(DeltaX*XInverse)) / 2.0;
	std::cout << "here8.3" << std::endl;
			Eigen::VectorXd AStarDeltaZ = Eigen::VectorXd::Zero(n);
			for (int i=0; i<n; i++) {
				AStarDeltaZ(i) = As[i].cwiseProduct(delta.Z).sum();
			}
	std::cout << "here8.4" << std::endl;
			delta.y = A_0.transpose().inverse() * (G*delta.x - AStarDeltaZ + delLCached);

	std::cout << "here9" << std::endl;

			// Calculate optimal step size using a line search TODO
			double alphaBarX = 1.0;
			double alphaBarZ = 1.0;
			alphaBarX = -gamma / smallestEigenvalue(XInverse * X(delta.x));
			alphaBarZ = -gamma / smallestEigenvalue(w.Z.inverse() * delta.Z);
			double alphaBar = std::min(std::min(alphaBarX, alphaBarZ), 1.0);
			int l = 2;
			double alpha = alphaBar * std::pow(beta, l);

	std::cout << "here10" << std::endl;

			// Update variables
			prevx = w.x;
			w.x += alpha*delta.x;
			w.y += delta.y;
			w.Z += alpha*delta.Z;
			
			// Inner-iteration output
			std::cout << "" << std::endl;
			std::cout << "      --------------------------------" << std::endl;
			std::cout << "           Inner Iteration " << k2 << std::endl;;
			std::cout << "      --------------------------------" << std::endl;
			std::cout << "             f(x) = " << f(w.x) << " <= " << maxVal << std::endl;;
			std::cout << "       rMag(w,mu) = " << rMagMu << " ?< " << epsilonPrime << std::endl;;
			std::cout << "                l = " << l << std::endl;;
			std::cout << "            alpha = " << alpha << std::endl;;
			
		}

		// Outer-iteration output
		std::cout << "" << std::endl;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "     Outer Iteration " << k << std::endl;;
		std::cout << "--------------------------------" << std::endl;
		std::cout << "       f(x) = " << f(w.x) << " <= " << maxVal << std::endl;;
		std::cout << "         mu = " << mu << std::endl;;
		std::cout << "  rMag(w,0) = " << rMagZero << " ?< " << epsilonPrime << std::endl;;
		
		// Update mu
		mu = mu / 10.0;

	}

	// Stop the timer
	auto t2 = std::chrono::high_resolution_clock::now();

	// Final output
	std::cout << "" << std::endl;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "      Final Output " << std::endl;;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "  iterations = " << k << std::endl;;
	std::cout << " time needed = " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << " s" << std::endl;
	std::cout << "  final f(x) = " << f(w.x) << " < " << maxVal << std::endl;;

	// Everything went fine
	return 0;

}

