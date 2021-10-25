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

// Local imports
#include "./data.h"
#include "./prettyPrint.h"

// Convert from a std::vector of vectors to an Eigen array
Eigen::MatrixXcd stdToEigen(std::vector<std::vector<std::complex<double>>> data) {
	Eigen::MatrixXcd toReturn = Eigen::MatrixXcd::Zero(data.size(), data[0].size());
	for (int i=0; i<data.size(); i++) {
		for (int j=0; j<data[0].size(); j++) {
			toReturn(i,j) = data[i][j];
		}
	}
	return toReturn;
}

// Standard cpp entry point
int main(int argc, char ** argv) {

	// Defining the problem
	int s = 2;
	int d = 2;

	// Loop over the command-line arguments
	for (int i=1; i<argc; i++){

		// Convert the char array to a standard string for easier processing
		std::string arg = argv[i];

		// If asking for help
		if (arg == "-h" || arg == "--help") {
			std::cout << "" << std::endl;
			std::cout << "---------------------------------" << std::endl;
			std::cout << "  Program that checks for MUBs" << std::endl;
			std::cout << "  using an SDP solver of a QCQP" << std::endl;
			std::cout << "---------------------------------" << std::endl;
			std::cout << "                        " << std::endl;
			std::cout << "       main options          " << std::endl;
			std::cout << " -h               show the help" << std::endl;
			std::cout << " -d [int]         set the dimension" << std::endl;
			std::cout << " -n [int]         set the number of measurements" << std::endl;
			std::cout << "" << std::endl;
			return 0;

		// Set the number of measurements 
		} else if (arg == "-d") {
			d = std::stoi(argv[i+1]);
			i += 1;

		// Use the BFGS update method
		} else if (arg == "-n") {
			s = std::stoi(argv[i+1]);
			i += 1;

		}

	}

	// Useful quantities
	int numPerm = s*(s-1)/2;
	int numMeasureB = s;
	int numOutcomeB = d;
	int numRealPer = (d*(d+1))/2-1;
	int numImagPer = (d*(d+1))/2-d;
	int numUniquePer = numRealPer + numImagPer;
	int numRhoMats = numPerm*numOutcomeB*numOutcomeB;
	int numBMats = numMeasureB*(numOutcomeB-1);
	int numMats = numRhoMats + numBMats;
	int N = numRhoMats + numMeasureB*numOutcomeB;
	int n = numMats*numUniquePer;
	int m = 1;

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
	int locRho = 0;
	std::vector<Eigen::Triplet<double>> tripsA_0;

	// For each pair of measurements
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {

			// For each of the B_i
			for (int k=0; k<numOutcomeB; k++) {

				// For each of the B_j
				for (int l=0; l<numOutcomeB; l++) {

					// rho * B^i_k
					if (k < numOutcomeB-1) {
						int locBik = (numRhoMats + i*(numOutcomeB-1) + k)*numUniquePer;
						for (int a=0; a<numUniquePer; a++) {
							tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBik+a, 2));
							tripsA_0.push_back(Eigen::Triplet<double>(locBik+a, locRho+a, 2));
						}
						for (int a=0; a<d-1; a++) {
							for (int b=0; b<d-1; b++) {
								if (a != b) {
									tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(a), locBik+diagLocs(b), 1));
									tripsA_0.push_back(Eigen::Triplet<double>(locBik+diagLocs(a), locRho+diagLocs(b), 1));
								}
							}
						}
						for (int a=0; a<d-1; a++) {
							b[0](locRho+diagLocs(a)) += -1;
							b[0](locBik+diagLocs(a)) += -1;
						}
						c[0] += 1;

					// For rho * (1-B^i-...)
					} else {
						for (int k=0; k<numOutcomeB-1; k++) {
							int locBik = (numRhoMats + i*(numOutcomeB-1) + k)*numUniquePer;
							for (int a=0; a<numUniquePer; a++) {
								tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBik+a, -2));
								tripsA_0.push_back(Eigen::Triplet<double>(locBik+a, locRho+a, -2));
							}
							for (int a=0; a<d-1; a++) {
								for (int b=0; b<d-1; b++) {
									if (a != b) {
										tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(a), locBik+diagLocs(b), -1));
										tripsA_0.push_back(Eigen::Triplet<double>(locBik+diagLocs(a), locRho+diagLocs(b), -1));
									}
								}
							}
							for (int a=0; a<d-1; a++) {
								b[0](locRho+diagLocs(a)) += 1;
								b[0](locBik+diagLocs(a)) += 1;
							}
						}
						c[0] += 2-d;

					}
					
					// rho * B^j_l
					if (l < numOutcomeB-1) {
						int locBjl = (numRhoMats + j*(numOutcomeB-1) + l)*numUniquePer;
						for (int a=0; a<numUniquePer; a++) {
							tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBjl+a, 2));
							tripsA_0.push_back(Eigen::Triplet<double>(locBjl+a, locRho+a, 2));
						}
						for (int a=0; a<d-1; a++) {
							for (int b=0; b<d-1; b++) {
								if (a != b) {
									tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(a), locBjl+diagLocs(b), 1));
									tripsA_0.push_back(Eigen::Triplet<double>(locBjl+diagLocs(a), locRho+diagLocs(b), 1));
								}
							}
						}
						for (int a=0; a<d-1; a++) {
							b[0](locRho+diagLocs(a)) += -1;
							b[0](locBjl+diagLocs(a)) += -1;
						}
						c[0] += 1;

					// For rho * (1-B^j-...)
					} else {
						for (int l=0; l<numOutcomeB-1; l++) {
							int locBjl = (numRhoMats + j*(numOutcomeB-1) + l)*numUniquePer;
							for (int a=0; a<numUniquePer; a++) {
								tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBjl+a, -2));
								tripsA_0.push_back(Eigen::Triplet<double>(locBjl+a, locRho+a, -2));
							}
							for (int a=0; a<d-1; a++) {
								for (int b=0; b<d-1; b++) {
									if (a != b) {
										tripsA_0.push_back(Eigen::Triplet<double>(locRho+diagLocs(a), locBjl+diagLocs(b), -1));
										tripsA_0.push_back(Eigen::Triplet<double>(locBjl+diagLocs(a), locRho+diagLocs(b), -1));
									}
								}
							}
							for (int a=0; a<d-1; a++) {
								b[0](locRho+diagLocs(a)) += 1;
								b[0](locBjl+diagLocs(a)) += 1;
							}
						}
						c[0] += 2-d;
					}

					// Next rho
					locRho += numUniquePer;

				}
				
			}

		}
	}

	// Create the matrix and 
	A[0].setFromTriplets(tripsA_0.begin(), tripsA_0.end());

	// Flip everything so it's a minimisation
	A[0] = -0.5*A[0];
	b[0] = -1*b[0];
	c[0] = -1*c[0];

	// Create mapping from mat loc to real vec loc
	Eigen::MatrixXi posToLocReal = Eigen::MatrixXi::Zero(d, d);
	int nextLoc = 0;
	int diagTerm = numUniquePer*10;
	for (int b=0; b<d-1; b++) {
		for (int a=b; a<d; a++) {
			posToLocReal(b, a) = nextLoc;
			posToLocReal(a, b) = nextLoc;
			nextLoc += 1;
		}
	}
	posToLocReal(d-1, d-1) = diagTerm;

	// Create mapping from mat loc to imag vec loc
	Eigen::MatrixXi posToLocImag = Eigen::MatrixXi::Zero(d, d);
	Eigen::MatrixXi imagSigns = Eigen::MatrixXi::Zero(d, d);
	for (int b=0; b<d-1; b++) {
		for (int a=b+1; a<d; a++) {
			posToLocImag(b, a) = nextLoc;
			posToLocImag(a, b) = nextLoc;
			imagSigns(b, a) = 1;
			imagSigns(a, b) = -1;
			nextLoc += 1;
		}
	}

	// Construct the constraint matrix
	std::vector<Eigen::Triplet<double>> tripsA_1;
	for (int i=0; i<numMeasureB; i++) {

		// For each of the B_i
		for (int k=0; k<numOutcomeB; k++) {

			// B^i_k * B^i_k
			if (k < numOutcomeB-1) {
				int locBik = (numRhoMats + i*(numOutcomeB-1) + k)*numUniquePer;
				for (int a=0; a<numUniquePer; a++) {
					tripsA_1.push_back(Eigen::Triplet<double>(locBik+a, locBik+a, 2));
				}
				for (int a=0; a<d-1; a++) {
					for (int b=0; b<d-1; b++) {
						if (a != b) {
							tripsA_1.push_back(Eigen::Triplet<double>(locBik+diagLocs(a), locBik+diagLocs(b), 1));
						}
					}
				}
				for (int a=0; a<d-1; a++) {
					b[1](locBik+diagLocs(a)) -= 2;
				}
				c[1] += 1;

			// For (1-B^i-...) * (1-B^i-...) TODO
			} else {
				for (int k=0; k<numOutcomeB-1; k++) {
					int locBik = (numRhoMats + i*(numOutcomeB-1) + k)*numUniquePer;
					for (int a=0; a<numUniquePer; a++) {
						tripsA_1.push_back(Eigen::Triplet<double>(locBik+a, locBik+a, 2));
					}
					for (int a=0; a<d-1; a++) {
						for (int b=0; b<d-1; b++) {
							if (a != b) {
								tripsA_1.push_back(Eigen::Triplet<double>(locBik+diagLocs(a), locBik+diagLocs(b), 1));
							}
						}
					}
					for (int a=0; a<d-1; a++) {
						//b[1](locBik+diagLocs(a)) -= 2+2*(2-d);
						b[1](locBik+diagLocs(a)) -= 2;
					}
					//for (int l=0; l<numOutcomeB-1; l++) {
						//if (k != l) {
							//int locOther = (numRhoMats + i*(numOutcomeB-1) + l)*numUniquePer;
							//for (int a=0; a<d-1; a++) {
								//for (int b=0; b<d-1; b++) {
									//if (a != b) {
										//tripsA_1.push_back(Eigen::Triplet<double>(locOther+diagLocs(a), locBik+diagLocs(b), 1));
										//tripsA_1.push_back(Eigen::Triplet<double>(locBik+diagLocs(a), locOther+diagLocs(b), 1));
									//}
								//}
							//}
						//}
					//}
				}
				//c[1] += d-1+std::pow(2-d, 2);
				c[1] += 1;

			}

		}

	}
	for (int i=0; i<numRhoMats; i++) {
		int locBik = i*numUniquePer;
		for (int a=0; a<numUniquePer; a++) {
			tripsA_1.push_back(Eigen::Triplet<double>(locBik+a, locBik+a, 2));
		}
		for (int a=0; a<d-1; a++) {
			for (int b=0; b<d-1; b++) {
				if (a != b) {
					tripsA_1.push_back(Eigen::Triplet<double>(locBik+diagLocs(a), locBik+diagLocs(b), 1));
				}
			}
		}
		for (int a=0; a<d-1; a++) {
			b[1](locBik+diagLocs(a)) -= 2;
		}
		c[1] += 1;
	}
	A[1].setFromTriplets(tripsA_1.begin(), tripsA_1.end());
	c[1] -= N;

	// Get the known results from the seesaw
	std::vector<std::vector<std::vector<std::complex<double>>>> Ms = getKnown(d, s);

	// Test with the ideal x
	Eigen::VectorXd xIdeal = Eigen::VectorXd::Zero(n);

	// Create the full matrix for debugging TODO
	Eigen::MatrixXd MIdeal = Eigen::MatrixXd::Zero(2*N*d, 2*N*d);
	int nextInd = 0;

	// Calculate the ideal rho's from these
	double rtd = std::sqrt(d);
	int rhoInd = 0;
	std::vector<std::vector<std::complex<double>>> rhos(numRhoMats);
	for (int i=0; i<numMeasureB; i++) {
		for (int j=i+1; j<numMeasureB; j++) {
			for (int k=0; k<numOutcomeB; k++) {
				for (int l=0; l<numOutcomeB; l++) {

					// The location in M for these B matrices
					int ind1 = i*numOutcomeB + k;
					int ind2 = j*numOutcomeB + l;

					// Create the rho
					Eigen::MatrixXcd B1 = stdToEigen(Ms[ind1]);
					Eigen::MatrixXcd B2 = stdToEigen(Ms[ind2]);
					Eigen::MatrixXcd newRho = B1 + B2 + rtd*(B1*B2 + B2*B1);
					newRho = newRho / newRho.trace();

					// TODO
					MIdeal.block(nextInd*2*d,   nextInd*2*d,   d, d) = newRho.real();
					MIdeal.block(nextInd*2*d+d, nextInd*2*d+d, d, d) = newRho.real();
					MIdeal.block(nextInd*2*d+d, nextInd*2*d,   d, d) = newRho.imag();
					MIdeal.block(nextInd*2*d,   nextInd*2*d+d, d, d) = -newRho.imag();
					nextInd++;

					// Put it into the vector
					for (int i1=0; i1<d-1; i1++) {
						for (int i2=i1; i2<d; i2++) {
							xIdeal(rhoInd+posToLocReal(i1, i2)) = std::real(newRho(i1, i2));
						}
					}
					for (int i1=0; i1<d-1; i1++) {
						for (int i2=i1+1; i2<d; i2++) {
							xIdeal(rhoInd+posToLocImag(i1, i2)) = std::imag(newRho(i1, i2));
						}
					}

					// Increment the rho
					rhoInd += numUniquePer;

				}
			}
		}
	}

	for (int i=0; i<numMeasureB; i++) {
		for (int k=0; k<numOutcomeB; k++) {
			int ind1 = i*numOutcomeB + k;
			Eigen::MatrixXcd B1 = stdToEigen(Ms[ind1]);
			MIdeal.block(nextInd*2*d,   nextInd*2*d,   d, d) = B1.real();
			MIdeal.block(nextInd*2*d+d, nextInd*2*d+d, d, d) = B1.real();
			MIdeal.block(nextInd*2*d+d, nextInd*2*d,   d, d) = B1.imag();
			MIdeal.block(nextInd*2*d,   nextInd*2*d+d, d, d) = -B1.imag();
			nextInd++;
		}
	}

	// Put the B's into the vector
	for (int i=0; i<numMeasureB; i++) {
		for (int k=0; k<numOutcomeB-1; k++) {
			int ind1 = i*numOutcomeB + k;
			Eigen::MatrixXcd B1 = stdToEigen(Ms[ind1]);
			int b1Ind = (i*(numOutcomeB-1)+k+numRhoMats)*numUniquePer;
			for (int i1=0; i1<d-1; i1++) {
				for (int i2=i1; i2<d; i2++) {
					xIdeal(b1Ind+posToLocReal(i1, i2)) = std::real(B1(i1, i2));
				}
			}
			for (int i1=0; i1<d-1; i1++) {
				for (int i2=i1+1; i2<d; i2++) {
					xIdeal(b1Ind+posToLocImag(i1, i2)) = std::imag(B1(i1, i2));
				}
			}
		}
	}

	// Test the ideal TODO
	double innerBound = xIdeal.dot(A[0]*xIdeal) + b[0].dot(xIdeal) + c[0];
	double con = xIdeal.dot(A[1]*xIdeal) + b[1].dot(xIdeal) + c[1];
	std::cout << xIdeal.dot(A[1]*xIdeal) << " " << b[1].dot(xIdeal) << " " << c[1] << std::endl;
	prettyPrint("x = ", xIdeal);
	prettyPrint("M = ", MIdeal);
	prettyPrint("tr(M) = ", MIdeal.trace());
	prettyPrint("tr(M^2) = ", (MIdeal*MIdeal).trace());
	prettyPrint("eigenvalues(M) = ", MIdeal.eigenvalues());
	prettyPrint("deter(M) = ", MIdeal.determinant());
	prettyPrint("2*d*N-1 = ", 2*d*N-1);
	prettyPrint("A_0 = ", A[0]);
	prettyPrint("b_0 = ", b[0]);
	prettyPrint("c_0 = ", c[0]);
	prettyPrint("A_1 = ", A[1]);
	prettyPrint("b_1 = ", b[1]);
	prettyPrint("c_1 = ", c[1]);
	prettyPrint("con of known = ", con);
	prettyPrint("obj of known = ", innerBound);

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
	mosek::fusion::Model::t model = new mosek::fusion::Model(); 
	auto _model = monty::finally([&](){model->dispose();});

	// The moment matrix to optimise
	auto dimX = monty::new_array_ptr(std::vector<int>({n, n}));
	mosek::fusion::Variable::t X = model->variable(dimX, mosek::fusion::Domain::inRange(-5.0, 5.0));
	mosek::fusion::Variable::t x = model->variable(n, mosek::fusion::Domain::inRange(-5.0, 5.0));

	// Set up the objective function 
	model->objective(mosek::fusion::ObjectiveSense::Minimize, mosek::fusion::Expr::add(mosek::fusion::Expr::dot(AMosek[0], X), mosek::fusion::Expr::dot(bMosek[0], x)));

	// X >= x^T x constraint
	model->constraint(
					mosek::fusion::Expr::vstack(
						mosek::fusion::Expr::hstack(X, x), 
						mosek::fusion::Expr::hstack(x->transpose(), 1)
					), 
					mosek::fusion::Domain::inPSDCone(n+1));

	// Other constraint
	model->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(AMosek[1], X), mosek::fusion::Expr::dot(bMosek[1], x)), mosek::fusion::Domain::equalsTo(-c[1]));

	// Solve the problem
	model->solve();

	// Extract the results
	double outerBound = model->primalObjValue() + c[0];
	auto temp = *(x->level());
	auto temp2 = *(X->level());
	Eigen::VectorXd xOpt = Eigen::VectorXd::Zero(n);
	Eigen::MatrixXd XOpt = Eigen::MatrixXd::Zero(n, n);
	for (int i=0; i<n; i++){
		xOpt(i) = temp[i];
	}
	for (int i=0; i<n*n; i++){
		XOpt(i/n,i%n) = temp2[i];
	}

	// TODO check
	std::cout << "output x con = " << XOpt.cwiseProduct(A[1]).sum() + b[1].dot(xOpt) + c[1] << std::endl;
	std::cout << "output x obj = " << XOpt.cwiseProduct(A[0]).sum() + b[0].dot(xOpt) + c[0] << std::endl;

	// Ouput the final results
	double allowedDiff = std::pow(10, -precision);
	prettyPrint("      outer bound = ", outerBound);
	prettyPrint(" best inner bound = ", innerBound);
	prettyPrint("   value for MUBs = ", criticalValue);
	if (outerBound > criticalValue + allowedDiff) {
		std::cout << "conclusion: there is no set of " << s << " MUBs in dimension " << d << std::endl;
	} else if (innerBound <= criticalValue + allowedDiff) {
		std::cout << "conclusion: there is a set of " << s << " MUBs in dimension " << d << std::endl;
	} else {
		std::cout << "conclusion: there might be a set of " << s << " MUBs in dimension " << d << std::endl;
	}
	std::cout << "            within an error of 1e-" << precision << "" << std::endl;

}

