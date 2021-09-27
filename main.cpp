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
	std::cout << std::scientific << pre << val << std::endl;
}
// Standard cpp entry point
int main(int argc, char ** argv) {

	int n = 18;
	int m = 6;

	// Create A_i
	std::vector<Eigen::SparseMatrix<double>> A(1+m, Eigen::SparseMatrix<double>(n, n));

	std::vector<Eigen::Triplet<double>> tripsA_0;
	tripsA_0.push_back(Eigen::Triplet<double>(12, 0, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(13, 1, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(14, 2, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(0, 12, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(1, 13, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(2, 14, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(15, 0, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(16, 1, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(17, 2, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(0, 15, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(1, 16, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(2, 17, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(12, 3, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(13, 4, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(14, 5, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(3, 12, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(4, 13, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(5, 14, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(15, 3, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(16, 4, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(17, 5, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(3, 15, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(4, 16, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(5, 17, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(12, 6, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(13, 7, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(14, 8, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(6, 12, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(7, 13, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(8, 14, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(15, 6, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(16, 7, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(17, 8, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(6, 15, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(7, 16, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(8, 17, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(12, 9, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(13, 10, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(14, 11, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(9, 12, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(10, 13, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(11, 14, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(15, 9, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(16, 10, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(17, 11, 1));
	tripsA_0.push_back(Eigen::Triplet<double>(9, 15, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(10, 16, -1));
	tripsA_0.push_back(Eigen::Triplet<double>(11, 17, 1));
	A[0].setFromTriplets(tripsA_0.begin(), tripsA_0.end());
	A[0] = -A[0];

	std::vector<Eigen::Triplet<double>> tripsA_1;
	tripsA_1.push_back(Eigen::Triplet<double>(0, 0, 1));
	tripsA_1.push_back(Eigen::Triplet<double>(1, 1, 1));
	tripsA_1.push_back(Eigen::Triplet<double>(2, 2, 1));
	A[1].setFromTriplets(tripsA_1.begin(), tripsA_1.end());

	std::vector<Eigen::Triplet<double>> tripsA_2;
	tripsA_2.push_back(Eigen::Triplet<double>(3, 3, 1));
	tripsA_2.push_back(Eigen::Triplet<double>(4, 4, 1));
	tripsA_2.push_back(Eigen::Triplet<double>(5, 5, 1));
	A[2].setFromTriplets(tripsA_2.begin(), tripsA_2.end());

	std::vector<Eigen::Triplet<double>> tripsA_3;
	tripsA_3.push_back(Eigen::Triplet<double>(6, 6, 1));
	tripsA_3.push_back(Eigen::Triplet<double>(7, 7, 1));
	tripsA_3.push_back(Eigen::Triplet<double>(8, 8, 1));
	A[3].setFromTriplets(tripsA_3.begin(), tripsA_3.end());

	std::vector<Eigen::Triplet<double>> tripsA_4;
	tripsA_4.push_back(Eigen::Triplet<double>(9, 9, 1));
	tripsA_4.push_back(Eigen::Triplet<double>(10, 10, 1));
	tripsA_4.push_back(Eigen::Triplet<double>(11, 11, 1));
	A[4].setFromTriplets(tripsA_4.begin(), tripsA_4.end());

	std::vector<Eigen::Triplet<double>> tripsA_5;
	tripsA_5.push_back(Eigen::Triplet<double>(12, 12, 1));
	tripsA_5.push_back(Eigen::Triplet<double>(13, 13, 1));
	tripsA_5.push_back(Eigen::Triplet<double>(14, 14, 1));
	A[5].setFromTriplets(tripsA_5.begin(), tripsA_5.end());

	std::vector<Eigen::Triplet<double>> tripsA_6;
	tripsA_6.push_back(Eigen::Triplet<double>(15, 15, 1));
	tripsA_6.push_back(Eigen::Triplet<double>(16, 16, 1));
	tripsA_6.push_back(Eigen::Triplet<double>(17, 17, 1));
	A[6].setFromTriplets(tripsA_6.begin(), tripsA_6.end());

	// Create b_i
	std::vector<Eigen::VectorXd> b(1+m, Eigen::VectorXd::Zero(n));
	b[0](0) = 2;
	b[0](9) = -2;
	b[1](0) = -1;
	b[2](3) = -1;
	b[3](6) = -1;
	b[4](9) = -1;
	b[5](12) = -1;
	b[6](15) = -1;

	// Create c_i
	std::vector<double> c(1+m, 0.0);
	c[0] = -4;

	// Ideal should be -6.82840 for d2n2

	// Test with the ideal x TODO
	Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
	//x << 0.8536,  0.3536, 0,
		 //0.8536, -0.3536, 0,
		 //0.1464,  0.3536, 0,
		 //0.1464, -0.3536, 0,
			  //1,       0, 0,
			//0.5,     0.5, 0;
	x << 1,  0, 0,
		 1,  0, 0,
		 0,  0, 0,
		 0, 0, 0,
		      1,       0, 0,
		    1,     0, 0; // TODO this equals 6 as it should


	prettyPrint("A_0 = ", A[0]);
	prettyPrint("x = ", x);

	double val = x.dot(A[0]*x) + b[0].dot(x) + c[0];
	prettyPrint("obj = ", val);
	prettyPrint("diff = ", val+6.82840);
	for (int i=1; i<1+m; i++) {
		val = x.dot(A[i]*x) + b[i].dot(x) + c[i];
		prettyPrint("con = ", val);
	}

}

