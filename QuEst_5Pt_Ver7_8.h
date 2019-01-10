#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
// includes to call Eigen
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <opencv2/opencv.hpp>
/*
   @Kaveh Fathiana, Jingfu Jina,
   Camera Relative Pose Estimation for Visual Servoing using Quaternions
*/

class QuEst 
{
public:
	QuEst() {}
	~QuEst(){}

	Eigen::MatrixXd solve_rotation(std::vector<Eigen::Vector3d>point_query, 
		std::vector<Eigen::Vector3d>point_train);

	Eigen::MatrixXd  CoefsVer(std::vector<Eigen::Vector3d>point_query,
		std::vector<Eigen::Vector3d>point_train);

	Eigen::MatrixXd coefsNumVer2_0(const Eigen::VectorXd &_mx1, const Eigen::VectorXd &_mx2,
		const Eigen::VectorXd &_my1,const Eigen::VectorXd &_my2, const Eigen::VectorXd &_nx2,
		const Eigen::VectorXd &_ny2,const Eigen::VectorXd &_r2, const Eigen::VectorXd &_s1,
		const Eigen::VectorXd &_s2);

	Eigen::MatrixXd coefsDenVer2_0(const Eigen::VectorXd &_mx2, const Eigen::VectorXd &_my2,
		const Eigen::VectorXd &_nx1, const Eigen::VectorXd &_nx2, const Eigen::VectorXd &_ny1,
		const Eigen::VectorXd &_ny2,const Eigen::VectorXd &_r1, const Eigen::VectorXd &_r2,
		const Eigen::VectorXd &_s2);

	Eigen::MatrixXd coefsNumDen(const Eigen::MatrixXd &_a1, const Eigen::MatrixXd &_a2, const
		Eigen::MatrixXd &_a3, const Eigen::MatrixXd &_a4, const Eigen::MatrixXd &_a5, const
		Eigen::MatrixXd &_a6, const Eigen::MatrixXd &_a7, const Eigen::MatrixXd &_a8, const
		Eigen::MatrixXd &_a9, const Eigen::MatrixXd &_a10, const Eigen::MatrixXd &_b1, const
		Eigen::MatrixXd &_b2, Eigen::MatrixXd &_b3, const Eigen::MatrixXd &_b4, const
		Eigen::MatrixXd &_b5, const	Eigen::MatrixXd &_b6, const Eigen::MatrixXd &_b7,
		const Eigen::MatrixXd &_b8, const Eigen::MatrixXd &_b9, const Eigen::MatrixXd &_b10);

	Eigen::MatrixXd coefsNumDen(const Eigen::VectorXd &_a1, const Eigen::VectorXd &_a2, const
		Eigen::VectorXd &_a3, const Eigen::VectorXd &_a4, const Eigen::VectorXd &_a5, const
		Eigen::VectorXd &_a6, const Eigen::VectorXd &_a7, const Eigen::VectorXd &_a8, const
		Eigen::VectorXd &_a9, const Eigen::VectorXd &_a10, const Eigen::VectorXd &_b1, const
		Eigen::VectorXd &_b2, Eigen::VectorXd &_b3, const Eigen::VectorXd &_b4, const
		Eigen::VectorXd &_b5, const	Eigen::MatrixXd &_b6, const Eigen::VectorXd &_b7,
		const Eigen::VectorXd &_b8, const Eigen::VectorXd &_b9, const Eigen::VectorXd &_b10);
	int nchoosek(const int n, const int k);

	inline double sgn(double val) { return (0.0 < val) - (val < 0.0); }
public:
	std::vector<Eigen::Vector3d> _point_query, _point_train;
};

Eigen::MatrixXd QuEst::solve_rotation(std::vector<Eigen::Vector3d>point_query,
	std::vector<Eigen::Vector3d>point_train)
{
	int Idx[][35] = { { 1, 2, 5, 11, 21, 3, 6, 12, 22, 8, 14, 24, 17, 27, 31, 4,
		7, 13, 23, 9, 15,25,18, 28, 32, 10, 16, 26, 19, 29, 33, 20, 30, 34, 35 },
					  { 2, 5, 11, 21, 36, 6, 12, 22, 37, 14, 24, 39, 27, 42, 46,
		7, 13, 23, 38, 15, 25, 40, 28, 43, 47, 16, 26, 41, 29, 44 , 48, 30, 45, 49, 50},
	{ 3, 6, 12, 22, 37, 8, 14, 24, 39, 17, 27, 42, 31, 46, 51, 9, 15, 25, 40, 18, 28, 43,
		32, 47, 52, 19, 29, 44, 33,	48, 53, 34, 49, 54, 55},
	{ 4, 7, 13, 23, 38, 9, 15, 25, 40, 18, 28, 43, 32, 47, 52, 10, 16, 26, 41, 19, 29, 44,
	  33, 48, 53, 20, 30, 45, 34, 49, 54, 35, 50, 55, 56 } };

	Eigen::MatrixXd  Cf = CoefsVer(point_query, point_train);
	
	int numEq = Cf.rows();
	// A is the coefficient matrix such that A * X = 0

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4 * numEq, 56);
	for (int i = 1; i <= 4; i++)
	{
		for (int j = 0; j < 35; j++)
		{
			int idx = Idx[i-1][j]-1;
			for (int k = (i - 1)*numEq, ncount_k = 0; k < i * numEq; k++, ncount_k++)
			{
				A(k, idx) = Cf(ncount_k, j);
			}
		}
	}
	Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	svd.compute(A,Eigen::ComputeThinV);

	Eigen::MatrixXd V = svd.matrixV();
	Eigen::MatrixXd N = V.block<56, 20>(0, 36);

	Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(35,20);
	Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(35, 20);
	//Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(35, 20);
	//Eigen::MatrixXd A3 = Eigen::MatrixXd::Zero(35, 20);
	for (int i = 0; i < 35; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			A0(i, j) = N(Idx[0][i]-1, j);
			A1(i, j) = N(Idx[1][i]-1, j);
			//A2(i, j) = N(Idx[2][i], j);
			//A3(i, j) = N(Idx[3][i], j);
		}
	}

	//Eigen::MatrixXd A234 = Eigen::MatrixXd::Zero(35, 20*3);
	//A234.block<35, 20>(0, 0) = A1;
	//A234.block<35, 20>(0, 20) = A2;
	//A234.block<35, 20>(0, 40) = A3;
	Eigen::MatrixXd A0_2 = A0.transpose()*A0;
	Eigen::MatrixXd A00 = (A0_2.transpose() + A0_2) / 2;
	Eigen::MatrixXd A0_T_2 = A0.transpose()*A1;
	Eigen::MatrixXd B = A00.ldlt().solve(A0_T_2);

	Eigen::MatrixXd B1 = B.topRows(20);
	//Eigen::MatrixXd B3 = B.bottomRows(20);
	//Eigen::MatrixXd B2 = B.block<20, B.cols>(20, 0);

	Eigen::EigenSolver<Eigen::MatrixXd> es1(B1);
	Eigen::MatrixXd V1 = es1.pseudoEigenvectors();

	//Eigen::EigenSolver<Eigen::MatrixXd> es2(B2);
	//Eigen::MatrixXd V2 = es2.pseudoEigenvectors();

	//Eigen::EigenSolver<Eigen::MatrixXd> es3(B3);
	//Eigen::MatrixXd V3 = es3.pseudoEigenvectors();

	//Eigen::MatrixXd Ve = Eigen::MatrixXd::Zero(V1.rows, V1.cols * 3);
	//Ve.leftCols(V1.cols) = V1;
	//Ve.rightCols(V1.cols) = V3;
	//Ve.block<V1.rows, V1.cols>(0, V1.cols) = V2;
	
	//Eigen::MatrixXd Ve = V1;

	Eigen::MatrixXd X5 = N * V1;

	Eigen::VectorXd status = Eigen::VectorXd::Zero(X5.row(0).size());
	for (int i = 0; i < X5.row(0).size(); ++i)
	{
		status(i) = sgn(X5.row(0)[i]);
	}


	for (int i = 0; i < X5.rows(); i++)
	{
		Eigen::VectorXd temp = (X5.block<1, 20>(i, 0));
		X5.block<1, 20>(i, 0) = (temp.cwiseProduct(status)).eval();
	}

	Eigen::VectorXd w = Eigen::VectorXd::Zero(20);
	Eigen::VectorXd w4 = Eigen::VectorXd::Zero(20);
	for (int i = 0; i < w.size(); i++)
	{
		w(i) = pow(X5(0, i), 1.0/5);
		w4(i) = pow(w(i), 4);
	}
	Eigen::VectorXd temp = (X5.block<1, 20>(1, 0));
	Eigen::VectorXd x = temp.cwiseQuotient(w4);
	temp = (X5.block<1, 20>(2, 0));
	Eigen::VectorXd y = temp.cwiseQuotient(w4);
	temp=(X5.block<1, 20>(3, 0));
	Eigen::VectorXd z = temp.cwiseQuotient(w4);

	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, 20);
	Q.block<1, 20>(0, 0) = w;
	Q.block<1, 20>(1, 0) = x;
	Q.block<1, 20>(2, 0) = y;
	Q.block<1, 20>(3, 0) = z;

	Eigen::MatrixXd QNm = Eigen::MatrixXd::Zero(4, 20);
	for (int i = 0; i < 20; i++)
	{
		Eigen::Quaterniond q =Eigen::Quaterniond(Q.block<4, 1>(0, i)).normalized();
		QNm.block<4, 1>(0, i) =Eigen::Vector4d(q.x(),q.y(),q.z(),q.w());
	}
	return QNm;
}

Eigen::MatrixXd  QuEst::CoefsVer(std::vector<Eigen::Vector3d>point_query,
	std::vector<Eigen::Vector3d>point_train)
{
	int numPts = point_query.size();
	int mat_length = nchoosek(numPts, 2) - 1;
	/*

	int *mat_data = new int[numPts * mat_length];

	for (int i = 0; i < numPts * mat_length; i++)
	{
		mat_data[i] = 0;
	}
	
	Eigen::MatrixXi idxBin1 = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic,
		Eigen::Dynamic, Eigen::RowMajor>>(mat_data, 2, mat_length);
	*/

	Eigen::MatrixXi idxBin1 = Eigen::MatrixXi::Zero(2, mat_length);
	int counter = -1;
	for (int i = 1;i<=numPts-2;i++)
	{
		for (int j = i + 1; j <= numPts; j++)
		{
			counter++;
			idxBin1(0, counter) = i-1;
			idxBin1(1, counter) = j-1;
		}
	}

	Eigen::VectorXd mx1(mat_length);
	Eigen::VectorXd my1(mat_length);
	Eigen::VectorXd nx1(mat_length);
	Eigen::VectorXd ny1(mat_length);

	Eigen::VectorXd mx2(mat_length);
	Eigen::VectorXd my2(mat_length);
	Eigen::VectorXd nx2(mat_length);
	Eigen::VectorXd ny2(mat_length);

	Eigen::VectorXd s1(mat_length);
	Eigen::VectorXd s2(mat_length);
	Eigen::VectorXd r1(mat_length);
	Eigen::VectorXd r2(mat_length);

	for (int i = 0; i < mat_length; i++)
	{
		mx1[i] = point_query[idxBin1(0, i)][0];
		my1[i] = point_query[idxBin1(0, i)][1];

		nx1[i] = point_train[idxBin1(0, i)][0];
		ny1[i] = point_train[idxBin1(0, i)][1];

		mx2[i] = point_query[idxBin1(1, i)][0];
		my2[i] = point_query[idxBin1(1, i)][1];

		nx2[i] = point_train[idxBin1(1, i)][0];
		ny2[i] = point_train[idxBin1(1, i)][1];

		s1[i] = point_query[idxBin1(0, i)][2];
		s2[i] = point_query[idxBin1(1, i)][2];

		r1[i] = point_train[idxBin1(0, i)][2];
		r2[i] = point_train[idxBin1(1, i)][2];
	}
	/*
	double * p_mx1 = new double[mat_length];
	double * p_mx2 = new double[mat_length];
	double * p_my1 = new double[mat_length];
	double * p_my2 = new double[mat_length];

	double * p_nx1 = new double[mat_length];
	double * p_ny1 = new double[mat_length];
	double * p_nx2 = new double[mat_length];
	double * p_ny2 = new double[mat_length];

	double * p_s1 = new double[mat_length];
	double * p_s2 = new double[mat_length];
	double * p_r1 = new double[mat_length];
	double * p_r2 = new double[mat_length];

	for (int i = 0; i < mat_length; i++)
	{
	p_mx1[i] = point_query[idxBin1(0, i)][0];
	p_my1[i] = point_query[idxBin1(0, i)][1];

	p_nx1[i] = point_train[idxBin1(0, i)][0];
	p_ny1[i] = point_train[idxBin1(0, i)][1];

	p_mx2[i] = point_query[idxBin1(1, i)][1];
	p_my2[i] = point_query[idxBin1(1, i)][0];

	p_nx2[i] = point_train[idxBin1(1, i)][0];
	p_ny2[i] = point_train[idxBin1(1, i)][1];

	p_s1[i] = point_query[idxBin1(0, i)][3];
	p_s2[i] = point_query[idxBin1(1, i)][3];

	p_r1[i] = point_train[idxBin1(0, i)][3];
	p_r2[i] = point_train[idxBin1(1, i)][3];
	}
	Eigen::VectorXd mx1 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_mx1, 1, mat_length);

	Eigen::VectorXd my1 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_my1, 1, mat_length);

	Eigen::VectorXd nx1 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_nx1, 1, mat_length);

	Eigen::VectorXd ny1 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_ny1, 1, mat_length);

	Eigen::VectorXd mx2 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_mx2, 1, mat_length);

	Eigen::VectorXd my2 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_my2, 1, mat_length);

	Eigen::VectorXd nx2 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_nx2, 1, mat_length);

	Eigen::VectorXd ny2 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_ny2, 1, mat_length);

	Eigen::VectorXd s1 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_s1, 1, mat_length);

	Eigen::VectorXd s2 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_s2, 1, mat_length);

	Eigen::VectorXd r1 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_r1, 1, mat_length);

	Eigen::VectorXd r2 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
	Eigen::Dynamic, Eigen::RowMajor>>(p_r2, 1, mat_length);
	*/

	// coefsN = num1
	Eigen::MatrixXd coefsN = coefsNumVer2_0(mx1, mx2, my1, my2, nx2, ny2, r2, s1, s2);

	//coefsD = den1
	Eigen::MatrixXd coefsD = coefsDenVer2_0(mx2, my2, nx1, nx2, ny1, ny2, r1, r2, s2);
	

	int numEq = nchoosek(numPts, 3);

	Eigen::MatrixXi idxBin2 = Eigen::MatrixXi::Zero(2, numEq);

	counter = -1;
	int counter2 = 0;

	for (int i=numPts-1;i>=2;i--)
	{
		for (int j = 1 + counter2; j <= i - 1 + counter2; j++)
		{
			for (int k = j + 1; k <= i + counter2; k++)
			{
				counter++;
				idxBin2(0, counter) = j - 1;
				idxBin2(1, counter) = k - 1;
			}
		}
		counter2 += i;
	}

	//ai = [num1; den1];
	Eigen::VectorXd a1 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a2 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a3 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a4 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a5 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a6 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a7 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a8 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a9 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd a10 = Eigen::VectorXd::Zero(2 * numEq);

	for (int i = 0; i < numEq; i++)
	{
		a1(i) = coefsN(idxBin2(0, i), 0);
		a1(i + numEq) = coefsD(idxBin2(0, i), 0);

		a2(i) = coefsN(idxBin2(0, i), 1);
		a2(i + numEq) = coefsD(idxBin2(0, i), 1);

		a3(i) = coefsN(idxBin2(0, i), 2);
		a3(i + numEq) = coefsD(idxBin2(0, i), 2);

		a4(i) = coefsN(idxBin2(0, i), 3);
		a4(i + numEq) = coefsD(idxBin2(0, i), 3);

		a5(i) = coefsN(idxBin2(0, i), 4);
		a5(i + numEq) = coefsD(idxBin2(0, i), 4);

		a6(i) = coefsN(idxBin2(0, i), 5);
		a6(i + numEq) = coefsD(idxBin2(0, i), 5);

		a7(i) = coefsN(idxBin2(0, i), 6);
		a7(i + numEq) = coefsD(idxBin2(0, i), 6);

		a8(i) = coefsN(idxBin2(0, i), 7);
		a8(i + numEq) = coefsD(idxBin2(0, i), 7);

		a9(i) = coefsN(idxBin2(0, i), 8);
		a9(i + numEq) = coefsD(idxBin2(0, i), 8);

		a10(i) = coefsN(idxBin2(0, i), 9);
		a10(i + numEq) = coefsD(idxBin2(0, i), 9);
	}

	//bi = [num2; den2];
	Eigen::VectorXd b1 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b2 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b3 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b4 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b5 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b6 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b7 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b8 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b9 = Eigen::VectorXd::Zero(2 * numEq);
	Eigen::VectorXd b10 = Eigen::VectorXd::Zero(2 * numEq);


	for (int i = 0; i < numEq; i++)
	{
		b1(i) = coefsD(idxBin2(1, i), 0);
		b1(i + numEq) = coefsN(idxBin2(1, i), 0);

		b2(i) = coefsD(idxBin2(1, i), 1);
		b2(i + numEq) = coefsN(idxBin2(1, i), 1);

		b3(i) = coefsD(idxBin2(1, i), 2);
		b3(i + numEq) = coefsN(idxBin2(1, i), 2);

		b4(i) = coefsD(idxBin2(1, i), 3);
		b4(i + numEq) = coefsN(idxBin2(1, i), 3);

		b5(i) = coefsD(idxBin2(1, i), 4);
		b5(i + numEq) = coefsN(idxBin2(1, i), 4);

		b6(i) = coefsD(idxBin2(1, i), 5);
		b6(i + numEq) = coefsN(idxBin2(1, i), 5);

		b7(i) = coefsD(idxBin2(1, i), 6);
		b7(i + numEq) = coefsN(idxBin2(1, i), 6);

		b8(i) = coefsD(idxBin2(1, i), 7);
		b8(i + numEq) = coefsN(idxBin2(1, i), 7);

		b9(i) = coefsD(idxBin2(1, i), 8);
		b9(i + numEq) = coefsN(idxBin2(1, i), 8);

		b10(i) = coefsD(idxBin2(1, i), 9);
		b10(i + numEq) = coefsN(idxBin2(1, i), 9);
	}

	// coefsND = [num1 * den2;den1 * num2];
	Eigen::MatrixXd coefsND = coefsNumDen(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
		b1, b2, b3, b4, b5, b6, b7, b8, b9, b10);
	//Matrix of all coefficients
	//coefs = (num1 * den2)  -  (den1 * num2)

	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(numEq, 35);

	C = coefsND.topRows(numEq) - coefsND.bottomRows(numEq);

	return C;
}

int QuEst::nchoosek(const int n, const int k)
{
	if (k > n / 2)
	{
		return nchoosek(n,n - k);
	}
	else if (k == 1)
	{
		return n;
	}
	else
	{
		double c = 1.0;
		for (int i = 1; i <= k; i++)
		{
			c *= (static_cast<double>(n) - k + i) / (static_cast<double>(i));
		}
		return static_cast<int>(std::round(c));
	}
}

Eigen::MatrixXd QuEst::coefsNumVer2_0(const Eigen::VectorXd &_mx1, const Eigen::VectorXd &_mx2,
	const Eigen::VectorXd &_my1, const Eigen::VectorXd &_my2, const Eigen::VectorXd &_nx2,
	const Eigen::VectorXd &_ny2, const Eigen::VectorXd &_r2, const Eigen::VectorXd &_s1,
	const Eigen::VectorXd &_s2)
{
	int length = _mx1.size();
	Eigen::VectorXd t2(length);
	Eigen::VectorXd t3(length);
	Eigen::VectorXd t4(length);
	Eigen::VectorXd t5(length);
	Eigen::VectorXd t6(length);
	Eigen::VectorXd t7(length);
	Eigen::VectorXd t8(length);
	Eigen::VectorXd t9(length);
	Eigen::VectorXd t10(length);
	Eigen::VectorXd t11(length);
	Eigen::VectorXd t12(length);
	Eigen::VectorXd t13(length);

	for (size_t i = 0; i < length; i++)
	{
		t2(i) = _mx1(i) * _my2(i) * _r2(i);
		t3(i) = _mx2(i) * _ny2(i) * _s1(i);
		t4(i) = _my1(i) * _nx2(i) * _s2(i);
		t5(i) = _mx1(i) * _nx2(i) * _s2(i) * 2.0;
		t6(i) = _my1(i) * _ny2(i) * _s2(i) * 2.0;
		t7(i) = _mx1(i) * _my2(i) * _nx2(i) * 2.0;
		t8(i) = _my2(i) * _r2(i) * _s1(i) * 2.0;
		t9(i) = _mx2(i) * _my1(i) * _r2(i);
		t10(i) = _mx1(i) * _ny2(i) * _s2(i);
		t11(i) = _mx2(i) * _my1(i) * _ny2(i) * 2.0;
		t12(i) = _mx2(i) * _r2(i) * _s1(i) * 2.0;
		t13(i) = _my2(i) *_nx2(i) * _s1(i);
	}

	Eigen::MatrixXd coefsN(length, 10);
	//double(*p_coefsN)[10] = new double[t2.size()][10];

	for (int i = 0; i < t2.size(); i++)
	{
		coefsN(i,0) = t2[i] + t3[i] + t4[i] - _mx2[i] * _my1[i] * 
			_r2[i] - _mx1[i] * _ny2[i] * _s2[i] - _my2[i] * _nx2[i] * _s1[i];

		coefsN(i,1) = t11[i] + t12[i] - _mx1[i] * _my2[i] * _ny2[i] * 2.0
			- _mx1[i] * _r2[i] * _s2[i] * 2.0;

		coefsN(i, 2) = t7[i] + t8[i] - _mx2[i] * _my1[i] * _nx2[i] * 2.0 -
			_my1[i] * _r2[i] * _s2[i] * 2.0;

		coefsN(i, 3) = t5[i] + t6[i] - _mx2[i] * _nx2[i] * _s1[i] * 2.0 -
			_my2[i] * _ny2[i] * _s1[i] * 2.0;

		coefsN(i, 4) = -t2[i] - t3[i] + t4[i] + t9[i] + t10[i] - _my2[i] *
			_nx2[i] * _s1[i];

		coefsN(i, 5) = -t5[i] + t6[i] + _mx2[i] * _nx2[i] * _s1[i] * 2.0 -
			_my2[i] * _ny2[i] * _s1[i] * 2.0;

		coefsN(i, 6) = t7[i] - t8[i] - _mx2[i] * _my1[i] * _nx2[i] * 2.0 +
			_my1[i] * _r2[i] * _s2[i] * 2.0;

		coefsN(i, 7) = -t2[i] + t3[i] - t4[i] + t9[i] - t10[i] + t13[i];

		coefsN(i, 8) = -t11[i] + t12[i] + _mx1[i] * _my2[i] * _ny2[i] * 2.0
			- _mx1[i] * _r2[i] * _s2[i] * 2.0;

		coefsN(i, 9) = t2[i] - t3[i] - t4[i] - t9[i] + t10[i] + t13[i];
	}

	return coefsN;
}

Eigen::MatrixXd QuEst::coefsDenVer2_0(const Eigen::VectorXd &_mx2, const Eigen::VectorXd &_my2,
	const Eigen::VectorXd &_nx1, const Eigen::VectorXd &_nx2, const Eigen::VectorXd &_ny1,
	const Eigen::VectorXd &_ny2, const Eigen::VectorXd &_r1, const Eigen::VectorXd &_r2,
	const Eigen::VectorXd &_s2)
{
	int length = _mx2.size();
	Eigen::VectorXd t2(length);
	Eigen::VectorXd t3(length);
	Eigen::VectorXd t4(length);
	Eigen::VectorXd t5(length);
	Eigen::VectorXd t6(length);
	Eigen::VectorXd t7(length);
	Eigen::VectorXd t8(length);
	Eigen::VectorXd t9(length);
	Eigen::VectorXd t10(length);
	Eigen::VectorXd t11(length);
	Eigen::VectorXd t12(length);
	Eigen::VectorXd t13(length);

	for (int i = 0; i < length; i++)
	{
		t2(i) = _mx2(i) * _ny1(i) * _r2(i);
		t3(i) = _my2(i) * _nx2(i) * _r1(i);
		t4(i) = _nx1(i) * _ny2(i) * _s2(i);
		t5(i) = _mx2(i) * _nx2(i) * _r1(i) * 2.0;
		t6(i) = _my2(i) * _ny2(i) * _r1(i) * 2.0;
		t7(i) = _mx2(i) * _nx2(i) * _ny1(i) * 2.0;
		t8(i) = _ny1(i) * _r2(i) * _s2(i) * 2.0;
		t9(i) = _my2(i) * _nx1(i) * _r2(i);
		t10(i) = _nx2(i) * _ny1(i) * _s2(i);
		t11(i) = _my2(i) * _nx1(i) * _ny2(i) * 2.0;
		t12(i) = _nx1(i) * _r2(i) * _s2(i) * 2.0;
		t13(i) = _mx2(i) *_ny2(i) * _r1(i);
	}

	Eigen::MatrixXd coefsN(length, 10);
	for (int i = 0; i < t2.size(); i++)
	{
		coefsN(i, 0) = t2[i] + t3[i] + t4[i] - _mx2[i] * _ny2[i] *_r1[i] - _my2[i] * 
			_nx1[i] * _r2[i] - _nx2[i] * _ny1[i] * _s2[i];

		coefsN(i, 1) = t11[i] + t12[i] - _my2[i] * _nx2[i] * _ny1[i] * 2.0
			- _nx2[i] * _r1[i] * _s2[i] * 2.0;

		coefsN(i, 2) = t7[i] + t8[i] - _mx2[i] * _nx1[i] * _ny2[i] * 2.0 -
			_ny2[i] * _r1[i] * _s2[i] * 2.0;

		coefsN(i, 3) = t5[i] + t6[i] - _mx2[i] * _nx1[i] * _r2[i] * 2.0 -
			_my2[i] * _ny1[i] * _r2[i] * 2.0;

		coefsN(i, 4) = t2[i] - t3[i] - t4[i] + t9[i] + t10[i] - _mx2[i] *
			_ny2[i] * _r1[i];

		coefsN(i, 5) = t5[i] - t6[i] - _mx2[i] * _nx1[i] * _r2[i] * 2.0 +
			_my2[i] * _ny1[i] * _r2[i] * 2.0;

		coefsN(i, 6) = -t7[i] + t8[i] + _mx2[i] * _nx1[i] * _ny2[i] * 2.0 -
			_ny2[i] * _r1[i] * _s2[i] * 2.0;

		coefsN(i, 7) = -t2[i] + t3[i] - t4[i] - t9[i] + t10[i] + t13[i];

		coefsN(i, 8) = t11[i] - t12[i] - _my2[i] * _nx2[i] * _ny1[i] * 2.0
			+ _nx2[i] * _r1[i] * _s2[i] * 2.0;

		coefsN(i, 9) = -t2[i] - t3[i] + t4[i] + t9[i] - t10[i] + t13[i];
	}

	return coefsN;
}

Eigen::MatrixXd QuEst::coefsNumDen(const Eigen::VectorXd &_a1, const Eigen::VectorXd &_a2, const
	Eigen::VectorXd &_a3, const Eigen::VectorXd &_a4, const Eigen::VectorXd &_a5, const
	Eigen::VectorXd &_a6, const Eigen::VectorXd &_a7, const Eigen::VectorXd &_a8, const
	Eigen::VectorXd &_a9, const Eigen::VectorXd &_a10, const Eigen::VectorXd &_b1, const
	Eigen::VectorXd &_b2, Eigen::VectorXd &_b3, const Eigen::VectorXd &_b4, const
	Eigen::VectorXd &_b5, const	Eigen::MatrixXd &_b6, const Eigen::VectorXd &_b7,
	const Eigen::VectorXd &_b8, const Eigen::VectorXd &_b9, const Eigen::VectorXd &_b10)
{
	int row_size = _a1.size();
	Eigen::MatrixXd M = Eigen::MatrixXd::Zero(row_size, 35);
	for (int i = 0; i < row_size; i++)
	{
		M(i, 0) = _a1(i)*_b1(i);
		M(i, 1) = _a1(i)*_b2(i) + _a2(i)*_b1(i);
		M(i, 2) = _a2(i)*_b2(i) + _a1(i)*_b5(i) + _a5(i)*_b1(i);
		M(i, 3) = _a2(i)*_b5(i) + _a5(i)*_b2(i);
		M(i, 4) = _a5(i)*_b5(i);
		M(i, 5) = _a1(i)*_b3(i) + _a3(i)*_b1(i);
		M(i, 6) = _a2(i)*_b3(i) + _a3(i)*_b2(i) + _a1(i)*_b6(i) + _a6(i)*_b1(i);
		M(i, 7) = _a2(i)*_b6(i) + _a3(i)*_b5(i) + _a5(i)*_b3(i) + _a6(i)*_b2(i);
		M(i, 8) = _a5(i)*_b6(i) + _a6(i)*_b5(i);
		M(i, 9) = _a3(i)*_b3(i) + _a1(i)*_b8(i) + _a8(i)*_b1(i);
		M(i, 10) = _a3(i)*_b6(i) + _a6(i)*_b3(i) + _a2(i)*_b8(i) + _a8(i)*_b2(i);
		M(i, 11) = _a6(i)*_b6(i) + _a5(i)*_b8(i) + _a8(i)*_b5(i);
		M(i, 12) = _a3(i)*_b8(i) + _a8(i)*_b3(i);
		M(i, 13) = _a6(i)*_b8(i) + _a8(i)*_b6(i);
		M(i, 14) = _a8(i)*_b8(i);
		M(i, 15) = _a1(i)*_b4(i) + _a4(i)*_b1(i);
		M(i, 16) = _a2(i)*_b4(i) + _a4(i)*_b2(i) + _a1(i)*_b7(i) + _a7(i)*_b1(i);
		M(i, 17) = _a2(i)*_b7(i) + _a4(i)*_b5(i) + _a5(i)*_b4(i) + _a7(i)*_b2(i);
		M(i, 18) = _a5(i)*_b7(i) + _a7(i)*_b5(i);
		M(i, 19) = _a3(i)*_b4(i) + _a4(i)*_b3(i) + _a1(i)*_b9(i) + _a9(i)*_b1(i);
		M(i, 20) = _a3(i)*_b7(i) + _a4(i)*_b6(i) + _a6(i)*_b4(i) + _a7(i)*_b3(i) + 
			_a2(i)*_b9(i) + _a9(i)*_b2(i);
		M(i, 21) = _a6(i)*_b7(i) + _a7(i)*_b6(i) + _a5(i)*_b9(i) + _a9(i)*_b5(i);
		M(i, 22) = _a3(i)*_b9(i) + _a4(i)*_b8(i) + _a8(i)*_b4(i) + _a9(i)*_b3(i);
		M(i, 23) = _a6(i)*_b9(i) + _a7(i)*_b8(i) + _a8(i)*_b7(i) + _a9(i)*_b6(i);
		M(i, 24) = _a8(i)*_b9(i) + _a9(i)*_b8(i);
		M(i, 25) = _a4(i)*_b4(i) + _a1(i)*_b10(i) + _a10(i)*_b1(i);
		M(i, 26) = _a4(i)*_b7(i) + _a7(i)*_b4(i) + _a2(i)*_b10(i) + _a10(i)*_b2(i);
		M(i, 27) = _a7(i)*_b7(i) + _a5(i)*_b10(i) + _a10(i)*_b5(i);
		M(i, 28) = _a3(i)*_b10(i) + _a4(i)*_b9(i) + _a9(i)*_b4(i) + _a10(i)*_b3(i);
		M(i, 29) = _a6(i)*_b10(i) + _a7(i)*_b9(i) + _a9(i)*_b7(i) + _a10(i)*_b6(i);
		M(i, 30) = _a8(i)*_b10(i) + _a9(i)*_b9(i) + _a10(i)*_b8(i);
		M(i, 31) = _a4(i)*_b10(i) + _a10(i)*_b4(i);
		M(i, 32) = _a7(i)*_b10(i) + _a10(i)*_b7(i);
		M(i, 33) = _a9(i)*_b10(i) + _a10(i)*_b9(i);
		M(i, 34) = _a10(i)*_b10(i);

	}
	return M;
}