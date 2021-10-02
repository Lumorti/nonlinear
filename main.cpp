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
int precision = 3;

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
	int n = numMats*numUniquePer;
	int m = 1 + numMats*2*(numRealPer+1);

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

					// rho * B^i_k TODO why is the imag not negative
					if (k < numOutcomeB-1) {
						int locBik = (numRhoMats + i*(numOutcomeB-1) + k)*numUniquePer;
						for (int a=0; a<numUniquePer; a++) {
							tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBik+a, 2));
							tripsA_0.push_back(Eigen::Triplet<double>(locBik+a, locRho+a, 2));
						}
						//for (int a=numRealPer; a<numUniquePer; a++) {
							//tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBik+a, -2));
							//tripsA_0.push_back(Eigen::Triplet<double>(locBik+a, locRho+a, -2));
						//}
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
							//for (int a=numRealPer; a<numUniquePer; a++) {
								//tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBik+a, 2));
								//tripsA_0.push_back(Eigen::Triplet<double>(locBik+a, locRho+a, 2));
							//}
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
						//for (int a=numRealPer; a<numUniquePer; a++) {
							//tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBjl+a, -2));
							//tripsA_0.push_back(Eigen::Triplet<double>(locBjl+a, locRho+a, -2));
						//}
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
							//for (int a=numRealPer; a<numUniquePer; a++) {
								//tripsA_0.push_back(Eigen::Triplet<double>(locRho+a, locBjl+a, 2));
								//tripsA_0.push_back(Eigen::Triplet<double>(locBjl+a, locRho+a, 2));
							//}
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

	// TODO check d3n2
	//prettyPrint("A_0 = ", A[0]);
	//prettyPrint("b_0 = ", b[0]);
	//prettyPrint("c_0 = ", c[0]);

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

	// Create the constraint matrices and vectors
	int nextInd = 1;
	for (int i=0; i<numMats; i++) {

		// Every row 
		for (int a1=0; a1<d; a1++) {

			// Times every column
			for (int a2=a1; a2<d; a2++) {

				// The vector of non-zero elements
				std::vector<Eigen::Triplet<double>> tripsAReal;
				std::vector<Eigen::Triplet<double>> tripsAImag;

				// Each pair within this multiplication
				int x1 = 0;
				int y1 = a1;
				int x2 = a2;
				int y2 = 0;
				for (int j=0; j<d; j++) {

					// If it's two originals (a+ib)(c+id) = ac-bd + i(bc+da)
					if (posToLocReal(y1,x1) != diagTerm && posToLocReal(y2,x2) != diagTerm) {

						// The real part
						tripsAReal.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocReal(y1,x1), i*numUniquePer+posToLocReal(y2,x2), 1));
						tripsAReal.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocImag(y1,x1), i*numUniquePer+posToLocImag(y2,x2), -imagSigns(y1,x1)*imagSigns(y2,x2)));

						// The imag part
						tripsAImag.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocReal(y1,x1), i*numUniquePer+posToLocImag(y2,x2), imagSigns(y2,x2)));
						tripsAImag.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocImag(y1,x1), i*numUniquePer+posToLocReal(y2,x2), imagSigns(y1,x1)));

					// If it's an original and a one-minus (a+ib)(1-c-d-...) = (a-ac-ad-...) + i(b-bc-bd-...)
					} else if (posToLocReal(y1,x1) != diagTerm && posToLocReal(y2,x2) == diagTerm) {

						// The real part
						b[nextInd](i*numUniquePer+posToLocReal(y1,x1)) += 1;
						for (int l=0; l<d-1; l++) {
							tripsAReal.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocReal(y1,x1), i*numUniquePer+posToLocReal(l,l), -1));
						}

						// The imag part
						b[nextInd+1](i*numUniquePer+posToLocImag(y1,x1)) += 1;
						for (int l=0; l<d-1; l++) {
							tripsAImag.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocImag(y1,x1), i*numUniquePer+posToLocReal(l,l), -imagSigns(y1,x1)));
						}

					// If it's a one-minus and an original (1-c-d-...)(a+ib) = (a-ac-ad-...) + i(b-bc-bd-...)
					} else if (posToLocReal(y1,x1) == diagTerm && posToLocReal(y2,x2) != diagTerm) {

						// The real part
						b[nextInd](i*numUniquePer+posToLocReal(y2,x2)) += 1;
						for (int l=0; l<d-1; l++) {
							tripsAReal.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocReal(y2,x2), i*numUniquePer+posToLocReal(l,l), -1));
						}

						// The imag part
						b[nextInd+1](i*numUniquePer+posToLocImag(y2,x2)) += 1;
						for (int l=0; l<d-1; l++) {
							tripsAImag.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocImag(y2,x2), i*numUniquePer+posToLocReal(l,l), -imagSigns(y2,x2)));
						}

					// If it's two one-minus (1-a-g-...)(1-c-e-...) = 1-a-g-c-e-...+ac+ae+gc+ge+...
					} else if (posToLocReal(y1,x1) == diagTerm && posToLocReal(y2,x2) == diagTerm) {

						// Only a real part
						c[nextInd] += 1;
						for (int l=0; l<d-1; l++) {
							b[nextInd](i*numUniquePer+posToLocReal(l,l)) -= 2;
						}
						for (int l1=0; l1<d-1; l1++) {
							for (int l2=0; l2<d-1; l2++) {
								tripsAReal.push_back(Eigen::Triplet<double>(i*numUniquePer+posToLocReal(l1,l1), i*numUniquePer+posToLocReal(l2,l2), 1));
							}
						}

					}

					// Next element
					x1 += 1;
					y2 += 1;

				}

				// Subtract just the real linear term if it's a one-minus
				if (posToLocReal(a1,a2) == diagTerm) {
					for (int l=0; l<d-1; l++) {
						b[nextInd](i*numUniquePer+posToLocReal(l,l)) += 1;
					}
					c[nextInd] -= 1;

				// Otherwise subtract both
				} else {
					b[nextInd](i*numUniquePer+posToLocReal(a1,a2)) -= 1;
					b[nextInd+1](i*numUniquePer+posToLocImag(a1,a2)) -= imagSigns(a1, a2);
				}

				// Construct the A matrices
				Eigen::SparseMatrix<double> newAReal(n, n);
				Eigen::SparseMatrix<double> newAImag(n, n);
				newAReal.setFromTriplets(tripsAReal.begin(), tripsAReal.end());
				newAImag.setFromTriplets(tripsAImag.begin(), tripsAImag.end());
				A[nextInd] = newAReal;
				A[nextInd+1] = newAImag;

				//A[nextInd] += Eigen::MatrixXd::Identity(n, n).sparseView();
				//A[nextInd+1] += Eigen::MatrixXd::Identity(n, n).sparseView();
				//c[nextInd] -= 3.5;
				//c[nextInd+1] -= 3.5;

				// TODO check d3n2
				//prettyPrint("A_i = ", A[nextInd]);
				//prettyPrint("b_i = ", b[nextInd]);
				//prettyPrint("c_i = ", c[nextInd]);
				//prettyPrint("A_i+1 = ", A[nextInd+1]);
				//prettyPrint("b_i+1 = ", b[nextInd+1]);
				//prettyPrint("c_i+1 = ", c[nextInd+1]);

				// We now have two more contraints
				nextInd += 2;

			}

		}

	}

	// Also need to add the identity-minus constraints TODO

	prettyPrint("m = ", m);
	prettyPrint("nextInd = ", nextInd);

	// TODO x^Tx = ? constraint
	//A[nextInd] += Eigen::MatrixXd::Identity(n, n).sparseView();
	//c[nextInd] -= 3.5;

	// Allow entry as the list of matrices
	std::vector<std::vector<std::vector<std::complex<double>>>> Ms(numMeasureB*numOutcomeB);

	// From the seesaw
	if (d == 2 && s == 2) {
		Ms[0] = { { 1.0, 0.0 }, 
				  { 0.0, 0.0 } };
		Ms[1] = { { 0.0, 0.0 }, 
				  { 0.0, 1.0 } };
		Ms[2] = { { 0.5, 0.5 }, 
				  { 0.5, 0.5 } };
		Ms[3] = { { 0.5,-0.5 }, 
				  {-0.5, 0.5 } };
	} else if (d == 2 && s == 3) {
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
	} else if (d == 2 && s == 4) {
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
	} else if (d == 3 && s == 2) {
		Ms[0] = { { +0.14489+0.00000i, +0.30540-0.13411i, -0.07569-0.08316i},
				  { +0.30540+0.13411i, +0.76785+0.00000i, -0.08256-0.24534i},
				  { -0.07569+0.08316i, -0.08256+0.24534i, +0.08727+0.00000i} };
		Ms[1] = { { +0.55013+0.00000i, -0.14651+0.21867i, -0.27624+0.31921i},
				  { -0.14651-0.21867i, +0.12594+0.00000i, +0.20045+0.02479i},
				  { -0.27624-0.31921i, +0.20045-0.02479i, +0.32393+0.00000i} };
		Ms[2] = { { +0.30498+0.00000i, -0.15888-0.08456i, +0.35192-0.23605i},
				  { -0.15888+0.08456i, +0.10622+0.00000i, -0.11789+0.22055i},
				  { +0.35192+0.23605i, -0.11789-0.22055i, +0.58880+0.00000i} };
		Ms[3] = { { +0.06823+0.00000i, +0.05579+0.11727i, +0.20946+0.05327i},
				  { +0.05579-0.11727i, +0.24715+0.00000i, +0.26282-0.31644i},
				  { +0.20946-0.05327i, +0.26282+0.31644i, +0.68461+0.00000i} };
		Ms[4] = { { +0.15982+0.00000i, +0.04739+0.28655i, -0.19954-0.10053i},
				  { +0.04739-0.28655i, +0.52782+0.00000i, -0.23942+0.32794i},
				  { -0.19954+0.10053i, -0.23942-0.32794i, +0.31236+0.00000i} };
		Ms[5] = { { +0.77195+0.00000i, -0.10318-0.40381i, -0.00993+0.04726i},
				  { -0.10318+0.40381i, +0.22503+0.00000i, -0.02339-0.01151i},
				  { -0.00993-0.04726i, -0.02339+0.01151i, +0.00302+0.00000i} };
	} else if (d == 3 && s == 5) {
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
	} else if (d == 4 && s == 4) {
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
	} else if (d == 6 && s == 4) {
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

	// Test with the ideal x
	Eigen::VectorXd xIdeal = Eigen::VectorXd::Zero(n);

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

	// Test the ideal TODO d3n2
	for (int i=1; i<1+m; i++) {
		double con = xIdeal.dot(A[i]*xIdeal) + b[i].dot(xIdeal) + c[i];
		prettyPrint("con = ", con);
	}
	prettyPrint("x = ", xIdeal);
	prettyPrint("x^Tx = ", xIdeal.dot(xIdeal));
	double innerBound = xIdeal.dot(A[0]*xIdeal) + b[0].dot(xIdeal) + c[0];
	prettyPrint("obj = ", innerBound);
	prettyPrint("crit = ", criticalValue);

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
	mosek::fusion::Variable::t X = model->variable(dimX, mosek::fusion::Domain::inRange(-2.0, 2.0));
	mosek::fusion::Variable::t x = model->variable(n, mosek::fusion::Domain::inRange(-2.0, 2.0));

	// Set up the objective function 
	model->objective(mosek::fusion::ObjectiveSense::Minimize, mosek::fusion::Expr::add(mosek::fusion::Expr::dot(AMosek[0], X), mosek::fusion::Expr::dot(bMosek[0], x)));

	// X >= x^T x constraint
	model->constraint(
					mosek::fusion::Expr::vstack(
						mosek::fusion::Expr::hstack(X, x), 
						mosek::fusion::Expr::hstack(x->transpose(), 1)
					), 
					mosek::fusion::Domain::inPSDCone(n+1));

	// Projectivity constraints
	for (int i=1; i<m+1; i++){
		model->constraint(mosek::fusion::Expr::add(mosek::fusion::Expr::dot(AMosek[i], X), mosek::fusion::Expr::dot(bMosek[i], x)), mosek::fusion::Domain::equalsTo(0.0));
	}

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

	// Ouput the final results
	double allowedDiff = std::pow(10, -precision);
	prettyPrint("      outer bound = ", outerBound);
	prettyPrint("known inner bound = ", innerBound);
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

