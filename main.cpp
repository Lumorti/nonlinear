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
#include <Eigen/../unsupported/Eigen/MatrixFunctions>

// MOSEK
#include "fusion.h"

// For printing
int precision = 4;

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
	std::cout << std::scientific << std::setprecision(precision) << pre << val << std::endl;
}
// Standard cpp entry point
int main(int argc, char ** argv) {

	// Defining the problem
	int s = 2;
	int d = 2;

	// Useful quantities
	int numPerm = sets*(sets-1)/2;
	int numMeasureB = s;
	int numOutcomeB = d;
	int numRealPer = (d*(d+1))/2-1;
	int numImagPer = (d*(d+1))/2-d;
	int numUniquePer = numRealPer + numImagPer;
	int numRhoMats = numPerm*numOutcomeB*numOutcomeB;
	int numBMats = numMeasureB*(numOutcomeB-1);
	int numMats = numRhoMats + numBMats;
	int n = numMats*numUniquePer;
	int m = 1 + numMats*(d*(d+1))/2;

	// Inner bound from seesaw/Monte Carlo/KKT 
	double innerBound = 0;
	if (d == 2 && s == 2) {
		innerBound = -6.82840;
	}

	// The value for MUBs
	double criticalValue = -numPerm*d*d*(1+(1/std::sqrt(d)));

	prettyPrint("n = ", n);
	prettyPrint("n*n = ", n*n);
	prettyPrint("m = ", m);

	// The location in the vector of the i'th diagonal
	Eigen::VectorXd diagLocs(d);
	int nextDelta = d;
	diagLocs(0) = 0;
	for (int i=1; i<d; i++) {
		diagLocs(i) = diagLocs(i-1) + nextDelta;
		nextDelta -= 1;
	}

	// Create the vector of As
	std::vector<Eigen::SparseMatrix<double>> A(1+m, Eigen::SparseMatrix<double>(n, n));

	// Create the vector of Bs
	std::vector<Eigen::VectorXd> b(1+m, Eigen::VectorXd::Zero(n));

	// Create the cs
	std::vector<double> c(1+m, 0.0);

	// Create the objective matrix and vector
	int rhoLoc = 0;
	std::vector<Eigen::Triplet<double>> tripsA_0;
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// The locations in the big matrix
					int locBik = numRhoMats*numUniquePer+(i*numOutcomeB+k)*numUniquePer;
					int locBjl = numRhoMats*numUniquePer+(i*numOutcomeB+k)*numUniquePer;

					// rho * B^i_k
					for (int a=0; a<numUniquePer; a++) {
						tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBik+a, 2));
						tripsA_0.push_back(Eigen::Triplet<double>(locBik+a, locRho+a, 2));
					}
					for (int a=0; a<d-1; a++) {
						for (int b=0; b<d-1; a++) {
							if (a != b) {
								tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(a), locBik+diagLocs(b), 1));
								tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(b), locBik+diagLocs(a), 1));
								tripsA_0.push_back(Eigen::Triplet<double>(locBik+diagLocs(a), locRho+diagLocs(b), 1));
								tripsA_0.push_back(Eigen::Triplet<double>(locBik+diagLocs(b), locRho+diagLocs(a), 1));
							}
						}
					}
					for (int a=0; a<d-1; a++) {
						b[0](locRho+diagLocs(a)) = -1;
						b[0](locBik+diagLocs(a)) = -1;
					}
					c[0] += 1;
					
					// rho * B^j_l
					for (int a=0; a<numUniquePer; a++) {
						tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBjl+a, 2));
						tripsA_0.push_back(Eigen::Triplet<double>(locBjl+a, locRho+a, 2));
					}
					for (int a=0; a<d-1; a++) {
						for (int b=0; b<d-1; a++) {
							if (a != b) {
								tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(a), locBjl+diagLocs(b), 1));
								tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(b), locBjl+diagLocs(a), 1));
								tripsA_0.push_back(Eigen::Triplet<double>(locBjl+diagLocs(a), locRho+diagLocs(b), 1));
								tripsA_0.push_back(Eigen::Triplet<double>(locBjl+diagLocs(b), locRho+diagLocs(a), 1));
							}
						}
					}
					for (int a=0; a<d-1; a++) {
						b[0](locRho+diagLocs(a)) = -1;
						b[0](locBjl+diagLocs(a)) = -1;
					}
					c[0] += 1;
					rhoLoc += numUniquePer;

				}

				// rho * (B^i_k + (1-B^j_...)) TODO

			}

			// rho * (B^j_l + (1-B^i_...)) TODO

		}
	}
	
	// Create the matrix and flip it
	A[0].setFromTriplets(tripsA_0.begin(), tripsA_0.end());
	A[0] = -0.5*A[0];

	// Create the constraint matrices and vectors TODO

	// Turn the Eigen A matrices into MOSEK forms
	std::vector<mosek::fusion::Matrix::t> AMosek(1+m);
	for (int j=0; j<1+m; j++){

		// Get the lists of locations and values
		std::vector<int> nonZeroRows;
		std::vector<int> nonZeroCols;
		std::vector<double> nonZeroVals;
		for (int i1=0; i1<A[j].outerSize(); ++i1){
			for (Eigen::SparseMatrix<double>::InnerIterator it(A[j], i1); it; ++it){
				nonZeroRows.push_back(it.row());
				nonZeroCols.push_back(it.col());
				nonZeroVals.push_back(it.value());
			}
		}

		// Make the sparse matrix from this data
		AMosek[j] = mosek::fusion::Matrix::sparse(n, n, monty::new_array_ptr(nonZeroRows), monty::new_array_ptr(nonZeroCols), monty::new_array_ptr(nonZeroVals));

	}

	// Turn the Eigen b vectors into MOSEK forms
	std::vector<mosek::fusion::Matrix::t> bMosek(1+m);
	for (int j=0; j<1+m; j++){

		// Get the lists of values
		std::vector<double> nonZeroVals(n);
		for (int i1=0; i1<b[j].size(); i1++){
			nonZeroVals[i1] = b[j](i1);
		}

		// Make the sparse matrix from this data
		bMosek[j] = mosek::fusion::Matrix::dense(n, 1, monty::new_array_ptr(nonZeroVals));

	}

	// Create the MOSEK model
	//mosek::fusion::Model::t model = new mosek::fusion::Model(); 
	//auto _model = monty::finally([&](){model->dispose();});

	//// The moment matrix to optimise
	//auto dimX = monty::new_array_ptr(std::vector<int>({n, n}));
	//mosek::fusion::Variable::t X = model->variable(dimX, mosek::fusion::Domain::inRange(-1.0, 1.0));
	//mosek::fusion::Variable::t x = model->variable(n, mosek::fusion::Domain::inRange(-1.0, 1.0));

	//// Set up the objective function 
	//model->objective(mosek::fusion::ObjectiveSense::Minimize, mosek::fusion::Expr::add(mosek::fusion::Expr::dot(AMosek[0], X), mosek::fusion::Expr::dot(bMosek[0], x)));

	//// X >= x^T x constraint
	//model->constraint(
					//mosek::fusion::Expr::vstack(
						//mosek::fusion::Expr::hstack(X, x), 
						//mosek::fusion::Expr::hstack(x->transpose(), 1)
					//), 
					//mosek::fusion::Domain::inPSDCone(n+1));

	//// Projectivity constraints
	//for (int i=1; i<m+1; i++){
		//model->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(AMosek[i], X), mosek::fusion::Expr::dot(bMosek[i], x)), mosek::fusion::Domain::equalsTo(0.0));
	//}

	//// Solve the problem
	//model->solve();

	//// Extract the results
	//double outerBound = model->primalObjValue() + c[0];
	//auto temp = *(x->level());
	//auto temp2 = *(X->level());
	//Eigen::VectorXd xOpt = Eigen::VectorXd::Zero(n);
	//Eigen::MatrixXd XOpt = Eigen::MatrixXd::Zero(n, n);
	//for (int i=0; i<n; i++){
		//xOpt(i) = temp[i];
	//}
	//for (int i=0; i<n*n; i++){
		//XOpt(i/n,i%n) = temp2[i];
	//}

	//// Ouput the final results
	//double allowedDiff = std::pow(10, -precision);
	//prettyPrint("      outer bound = ", outerBound);
	//prettyPrint("known inner bound = ", innerBound);
	//prettyPrint("   value for MUBs = ", criticalValue);
	//if (outerBound > criticalValue + allowedDiff) {
		//std::cout << "conclusion: there is no set of " << s << " MUBs in dimension " << d << std::endl;
	//} else if (innerBound <= criticalValue + allowedDiff) {
		//std::cout << "conclusion: there is a set of " << s << " MUBs in dimension " << d << std::endl;
	//} else {
		//std::cout << "conclusion: there might be a set of " << s << " MUBs in dimension " << d << std::endl;
	//}
		//std::cout << "            within an error of 1e-" << precision << "" << std::endl;

	// Test with the ideal x TODO
	xOpt << 0.8536,  0.3536, 0,
		 0.8536, -0.3536, 0,
		 0.1464,  0.3536, 0,
		 0.1464, -0.3536, 0,
			  1,       0, 0,
			0.5,     0.5, 0;
	prettyPrint("A_0 = ", A[0]);
	prettyPrint("b_0 = ", b[0]);
	prettyPrint("c_0 = ", c[0]);
	prettyPrint("x = ", xOpt);
	double obj = xOpt.dot(A[0]*xOpt) + b[0].dot(xOpt) + c[0];
	for (int i=1; i<1+m; i++) {
		obj = xOpt.dot(A[i]*xOpt) + b[i].dot(xOpt) + c[i];
		prettyPrint("con = ", obj);
	}

}

