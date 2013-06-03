#include <cassert>
#include <cmath>
#include <numeric>
#include <cv.h>
#include "../System/System.h"

namespace TurboCV
{
	namespace System
	{
		namespace ML
		{
			class SNE
			{
			public:
				template<typename T>
				cv::Mat Compute(ArrayList<ArrayList<T>> samples, double perplexity = 30.0, 
					int yDims = 2)
				{
					assert(samples.Count() > 0);
					int n = samples.Count(), xDims = samples[0].Count();

					cv::Mat X(n, xDims, CV_64F);
					for (int i = 0; i < n; i++)
						for (int j = 0; j < xDims; j++)
							X.at<double>(i, j) = samples[i][j];

					return Compute(X, perplexity, yDims);
				}

				cv::Mat Compute(cv::InputArray samples, double perplexity = 30.0, int yDims = 2)
				{
					cv::Mat X = samples.getMat().clone();
					normalizeVectors(X);

					const int n = X.rows;
					const double eps = 1e-12;
					const int MAX_ITER = 1000;
					const double INIT_MOMENTUM = 0.5;
					const double FINAL_MOMENTUM = 0.8;
					const double eta = 0.05;
					const double jitterDecay = 0.99;
					double jitter = 0.3;

					cv::Mat Y(n, yDims, CV_64F);
					cv::randn(Y, 0, 1);

					cv::Mat iY = cv::Mat::zeros(n, yDims, CV_64F);

					printf("Compute P...\n");
					cv::Mat P = getP(X, 1e-5, perplexity);
					P = cv::max(eps, P);

					for (int i = 0; i < MAX_ITER; i++)
					{
						cv::Mat D = getDistanceMatrix(Y);
						cv::Mat gaussD;
						cv::exp(-D, gaussD);
						for (int j = 0; j < n; j++)
							gaussD.at<double>(j, j) = 0;

						cv::Mat Q(gaussD.size(), CV_64F);
						for (int j = 0; j < Q.rows; j++)
						{
							double sum = 0;
							for (int k = 0; k < Q.cols; k++)
								sum += gaussD.at<double>(j, k);

							for (int k = 0; k < Q.cols; k++)
								Q.at<double>(j, k) = gaussD.at<double>(j, k) / sum;
						}
						Q = cv::max(eps, Q);

						cv::Mat PQ = P - Q;
						cv::Mat dY = cv::Mat::zeros(n, yDims, CV_64F);
						for (int j = 0; j < n; j++)
						{
							for (int k = 0; k < n; k++)
							{
								double tmp = PQ.at<double>(j, k) + PQ.at<double>(k, j);

								for (int m = 0; m < yDims; m++)
									dY.at<double>(j, m) += tmp *
										(Y.at<double>(j, m) - Y.at<double>(k, m));
							}
						}

						double momentum = i < 750? INIT_MOMENTUM : FINAL_MOMENTUM;
						iY = momentum * iY - eta * dY;

						Y += iY;

						cv::Mat normalDistribution(Y.size(), CV_64F);
						cv::randn(normalDistribution, 0, 1);
						Y += jitter * normalDistribution;
						jitter *= jitterDecay;

						normalizeVectors(Y);

						if ((i + 1) % 10 == 0)
						{
							cv::Mat tmp;
							cv::log(P / Q, tmp);
							double C = cv::sum(P.mul(tmp))[0];
							printf("Iteration %d: error is %f\n", i + 1, C);
						}
					}

					return Y;
				}

			private:
				static cv::Mat getDistanceMatrix(cv::InputArray matrix)
				{
					cv::Mat mat = matrix.getMat();
					int n = mat.rows, d = mat.cols;
					cv::Mat D = cv::Mat::zeros(n, n, CV_64F);

					#pragma omp parallel for
					for (int i = 0; i < n; i++)
					{
						for (int j = i + 1; j < n; j++)
						{
							double sum = 0;
							for (int k = 0; k < d; k++)
								sum += (mat.at<double>(i, k) - mat.at<double>(j, k)) *
								    (mat.at<double>(i, k) - mat.at<double>(j, k));

							D.at<double>(i, j) = D.at<double>(j, i) = sum;
						}
					}

					return D;
				}

				static cv::Mat normalizeVectors(cv::InputOutputArray matrix)
				{
					cv::Mat mat = matrix.getMat();

					cv::Mat center = cv::Mat::zeros(1, mat.cols, CV_64F);
					for (int i = 0; i < mat.rows; i++)
						for (int j = 0; j < mat.cols; j++)
							center.at<double>(0, j) += mat.at<double>(i, j);

					for (int i = 0; i < mat.cols; i++)
						center.at<double>(0, i) /= mat.rows;

					for (int i = 0; i < mat.rows; i++)
						for (int j = 0; j < mat.cols; j++)
							mat.at<double>(i, j) -= center.at<double>(0, j);

					return center;
				}

				static cv::Mat getP(cv::Mat X, double tolerance = 1e-5, double perplexity = 30.0)
				{
					int n = X.rows, d = X.cols;
					cv::Mat D = getDistanceMatrix(X);
					cv::Mat P = cv::Mat::zeros(n, n, CV_64F);
					ArrayList<double> sigmaSqr(n, 1.0);
					const double logU = std::log(perplexity);
					const double INF = 1e12;

					#pragma omp parallel for
					for (int i = 0; i < n; i++)
					{
						double lowerBound = -INF;
						double upperBound = INF;

						cv::Mat Di(1, n - 1, CV_64F);
						for (int j = 0; j < i; j++)
							Di.at<double>(0, j) = D.at<double>(i, j);
						for (int j = i + 1; j < n; j++)
							Di.at<double>(0, j - 1) = D.at<double>(i, j);

						cv::Mat Pi;
						double Hdiff = 0;
						int tries = 0;
						do 
						{
							cv::Mat gaussDi(Di.size(), CV_64F);
							cv::exp(Di / -sigmaSqr[i], gaussDi);
							double sumGDi = cv::sum(gaussDi)[0];

							double H = Di.dot(gaussDi) / (sumGDi * sigmaSqr[i]) + std::log(sumGDi);
							Hdiff = logU - H;

							Pi = gaussDi / sumGDi;

							if (Hdiff > 0)
							{
								lowerBound = sigmaSqr[i];

								if (upperBound == INF)
									sigmaSqr[i] *= 2;
								else
									sigmaSqr[i] = (sigmaSqr[i] + upperBound) / 2;
							}
							else
							{
								upperBound = sigmaSqr[i];

								if (lowerBound == -INF)
									sigmaSqr[i] /= 2;
								else
									sigmaSqr[i] = (sigmaSqr[i] + lowerBound) / 2;
							}

							tries++;
						} while (std::abs(Hdiff) > tolerance && tries < 50);

						for (int j = 0; j < i; j++)
							P.at<double>(i, j) = Pi.at<double>(0, j);
						for (int j = i + 1; j < n; j++)
							P.at<double>(i, j) = Pi.at<double>(0, j - 1);
					}

					return P;
				}
			};
		}
	}
}