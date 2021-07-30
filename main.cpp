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
std::vector<Eigen::MatrixXd> As;

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

// Pretty print a complex 2D sparse Eigen array
template <typename type>
void prettyPrint(std::string pre, Eigen::SparseMatrix<type> arr) {

	// Extract the dense array and then call the routine as normal
	prettyPrint(pre, Eigen::Matrix<type,-1,-1>(arr));

}

// Get the smallest eigenvalue of a matrix
double smallestEigenvalue(Eigen::MatrixXd A){

	// Start with a random unit vector
	Eigen::VectorXd w = Eigen::VectorXd::Random(A.cols());
	Eigen::VectorXd v = w.normalized();

	// We want the lowest not the highest
	Eigen::MatrixXd AInv = A.inverse();

	// Iteratively solve Aw = v, 
	double eigenval = 1.0 / w.norm();
	for (int i=0; i<10; i++){
		w = AInv * v;
		eigenval = 1.0 / w.norm();
		v = w.normalized();
	}

	// Return the minimum eigenvalue of A
	return eigenval;

}

// Objective function
double f(Eigen::VectorXd x) {

	// Init the return val
	double val = 0.0;

	// Off-diagonal elements should be doubled since it's symmetric
	std::vector<double> factors(numRealPer, 2.0);
	int ind = 0;
	int ref = 0;
	int diff = d;
	while (ind < numRealPer) {
		factors[ind] = 1.0;
		factors[ind+1] = 1.0;
		ind += 2*diff;
		diff--;
	}

	// For each pair of measurements
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
	
			// For each outcome of these measurements
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// Where in the vector the matrices start
					int startLoc1 = (i*numOutcomeB + k) * numRealPer;
					int startLoc2 = (j*numOutcomeB + l) * numRealPer;

					// Get the real component
					double realComp = 0.0;
					double imagComp = 0.0;
					for (int a=0; a<numRealPer; a+=2) {
						realComp += factors[a] * x(startLoc1+a)   * x(startLoc2+a);
						imagComp += factors[a] * x(startLoc1+a+1) * x(startLoc2+a+1);
					}

					// Add this inner product
					val += std::sqrt(1.0 - realComp + imagComp);

				}
			}

		}
	}

	// Return the function value
	return val;

}

// Gradient of the objective function
Eigen::VectorXd delF(Eigen::VectorXd x) {

	// Init the return val
	Eigen::VectorXd vals = Eigen::VectorXd::Zero(n);

	// Off-diagonal elements should be doubled since it's symmetric
	std::vector<double> factors(numRealPer, 2.0);
	int ind = 0;
	int ref = 0;
	int diff = d;
	while (ind < numRealPer) {
		factors[ind] = 1.0;
		factors[ind+1] = 1.0;
		ind += 2*diff;
		diff--;
	}

	// For each pair of measurements
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
	
			// For each outcome of these measurements
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// Where in the vector the matrices start
					int startLoc1 = (i*numOutcomeB + k) * numRealPer;
					int startLoc2 = (j*numOutcomeB + l) * numRealPer;

					// Get the real component
					double realComp = 0.0;
					double imagComp = 0.0;
					double realTop = 0.0;
					double imagTop = 0.0;
					for (int a=0; a<numRealPer; a+=2) {
						realComp += factors[a] * x(startLoc1+a)   * x(startLoc2+a);
						imagComp += factors[a] * x(startLoc1+a+1) * x(startLoc2+a+1);
						realTop += factors[a] * x(startLoc2+a);
						imagTop += factors[a] * x(startLoc2+a+1);
					}

					// For each component, add to that section of the differentiation
					for (int a=0; a<numRealPer; a++) {
						vals(startLoc1+a) -= x(startLoc1+a) / (2.0 * std::sqrt(1.0 - realComp + imagComp));
					}

				}
			}

		}
	}

	// Return the function value
	return vals;

}

// Gradient of the constraint function
Eigen::MatrixXd delG(Eigen::VectorXd x) {

	// Matrix representing the Jacobian of g
	Eigen::MatrixXd gOutput = Eigen::MatrixXd::Zero(m, n);

	// For each measurement
	for (int i=0; i<numMeasureB; i++) {
	
		// For each position
		for (int a=0; a<numRealPer; a++) {

			// For each outcome of this measurement
			for (int k=0; k<numOutcomeB; k++) {
				int vecLoc = (i*numOutcomeB + k) * numRealPer;
				gOutput(i*numRealPer + a, vecLoc + a) = 1;
			}

		}

	}

	// Return the gradient vector of g
	return gOutput;

}

// Differential of the Lagrangian given individual components
Eigen::VectorXd delL(Eigen::VectorXd x, Eigen::VectorXd y, Eigen::MatrixXd Z) {

	// Start with a vector of n elements
	Eigen::VectorXd vals = Eigen::VectorXd::Zero(n);

	// Get the gradient of f(x)
	Eigen::VectorXd delFCached = delF(x);

	// Get A_0
	Eigen::MatrixXd A_0 = delG(x);

	// Calculate the first part as in the paper
	vals = delFCached - A_0.transpose() * y;

	// Then the second 
	for (int i=0; i<n; i++) {
		vals(i) -= As[i].cwiseProduct(Z).sum();
	}

	// Return this vector
	return vals;

}

// Differential of the Lagrangian given an interior point
Eigen::VectorXd delL(interiorPoint w) {
	return delL(w.x, w.y, w.Z);
}

// Constraint function
Eigen::VectorXd g(Eigen::VectorXd x) {

	// Vector to create
	Eigen::VectorXd gOutput = Eigen::VectorXd::Zero(m);

	// The values the elements in x should sum to
	std::vector<double> ideal(numRealPer, 0.0);
	int ind = 0;
	int ref = 0;
	int diff = d;
	while (ind < numRealPer) {
		ideal[ind] = 1.0;
		ind += 2*diff;
		diff--;
	}

	// For each measurement
	for (int i=0; i<numMeasureB; i++) {
	
		// For each position
		for (int a=0; a<numRealPer; a++) {

			// For each outcome of this measurement
			for (int k=0; k<numOutcomeB; k++) {
				int vecLoc = (i*numOutcomeB + k) * numRealPer;
				gOutput(i*numRealPer + a) += x(vecLoc + a);
			}

			// Take away the ideal (thus should be zero here)
			gOutput(i*numRealPer + a) -= ideal[a];

		}

	}

	// Return this vector of things that should be zero
	return gOutput;

}

// Function turning x to X
Eigen::MatrixXd X(Eigen::VectorXd x) {

	// Create a blank p by p matrix
	Eigen::MatrixXd newX = Eigen::MatrixXd::Zero(p, p);

	// Cached quantities
	int halfP = p / 2;

	// For each vector element
	for (int i=0; i<n; i++) {
		newX += As[i] * x(i);
	}
	
	// Return this new matrix
	return newX;

}

// Function giving the norm of a point, modified by some mu
double rMag(interiorPoint w, double mu) {

	// Calculate various vectors
	Eigen::MatrixXd XCached = X(w.x);
	Eigen::VectorXd gCached = g(w.x);
	Eigen::VectorXd delLCached = delL(w);

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
	numOutcomeB = d;
	numUniquePer = (d*(d+1))/2;
	numRealPer = numUniquePer*2;

	// Sizes of matrices
	n = numMeasureB*numOutcomeB*numRealPer;
	m = numMeasureB*numRealPer;
	p = numMeasureB*numOutcomeB*d*2;

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
			int matLoc = (i*numOutcomeB + k) * d;
			int vecLoc = (i*numOutcomeB + k) * numRealPer;
			int nextX = 0;
			int nextY = 0;

			// Loop over these vector elements
			for (int l=0; l<numUniquePer; l++) {

				// Create two blank p by p matrices
				Eigen::MatrixXd newAReal = Eigen::MatrixXd::Zero(p, p);
				Eigen::MatrixXd newAImag = Eigen::MatrixXd::Zero(p, p);

				// Place the real comps in the diagonal blocks
				newAReal(matLoc+nextX, matLoc+nextY) = 1;
				newAReal(matLoc+nextY, matLoc+nextX) = 1;
				newAReal(matLoc+nextX+halfP, matLoc+nextY+halfP) = 1;
				newAReal(matLoc+nextY+halfP, matLoc+nextX+halfP) = 1;

				// Place the imag comps in the off-diagonal blocks
				newAImag(matLoc+nextX+halfP, matLoc+nextY) = 1;
				newAImag(matLoc+nextY, matLoc+nextX+halfP) = -1;
				newAImag(matLoc+nextX, matLoc+nextY+halfP) = -1;
				newAImag(matLoc+nextY+halfP, matLoc+nextX) = 1;

				// Move the location along
				nextX++;
				if (nextX >= d) {
					nextY++;
					nextX = nextY;
				}

				// Add these to the list
				As.push_back(newAReal);
				As.push_back(newAImag);

			}

		}

	}

	// The interior point to optimise
	interiorPoint w(n, m, p);

	// Initialise x
	w.x << 1.0, 0.0,    0.0, 0.0,     0.0, 0.0,
		   0.0, 0.0,    0.0, 0.0,     1.0, 0.0,
		   0.5, 0.0,    0.5, 0.0,     0.5, 0.0,
		   0.5, 0.0,   -0.5, 0.0,     0.5, 0.0;
	//w.x << 1.0, 0.0,    0.0, 0.0,     0.0, 0.0,
		   //0.0, 0.0,    0.0, 0.0,     1.0, 0.0,
		   //1.0, 0.0,    0.0, 0.0,     0.0, 0.0,
		   //0.0, 0.0,    0.0, 0.0,     1.0, 0.0;

	// Initialise Z
	w.Z = Eigen::MatrixXd::Identity(p, p);

	// Calculate the initial G, the Hessian of L(w) TODO
	Eigen::MatrixXd G = del2L(w);

	// Ensure G is positive semidefinite TODO
	Eigen::MatrixXd cholDecomp = G.ldlt();
			
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

			// Calculate T, the scaling matrix
			Eigen::MatrixXd T = XCached.pow(-0.5);

			// Update G
			Eigen::VectorXd q = delL(w) - delL(prevX, w.y, w.Z);
			Eigen::VectorXd s = w.x - prevx;
			Eigen::VectorXd sTran = s.transpose();
			double psi = 1;
			if (sTran*q < 0.2*sTran*(G*s)){
				psi = (0.8*sTran*(G*s)) / (sTran(G*s-q));
			}
			Eigen::VectorXd qBar = psi*q + (1-psi)*(G*s);
			G += -((G*(s*(sTran*G))) / (sTran*(G*s))) + ((qBar*qBar.transpose()) / (sTran*qBar));
			
			// Calculate direction TODO
			interiorPoint delta(n, m, p);

			// Calculate optimal step size using a line search TODO
			double alphaBarX = 1.0;
			double alphaBarZ = 1.0;
			if (XCached.determinant() != 0.0){
				alphaBarX = -gamma / smallestEigenvalue(XCached.inverse() * X(delta.x));
			}
			Eigen::MatrixXd eigenZ = w.Z.inverse() * delta.Z;
			if (eigenZ.determinant() != 0.0){
				alphaBarZ = -gamma / smallestEigenvalue(eigenZ);
			}
			double alphaBar = std::min(std::min(alphaBarX, alphaBarZ), 1.0);
			int l = 2;
			double alpha = alphaBar * std::pow(beta, l);

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

