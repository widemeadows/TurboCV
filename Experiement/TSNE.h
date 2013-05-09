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
        class TSNE
        {
        public:
            template<typename T>
            void Compute(ArrayList<ArrayList<T>> samples, int dims, double perplexity = 30.0)
            {
                assert(samples.Count() > 0);

				int n = samples.Count();
				int d = samples[0].Count();
				cv::Mat X(n, d, CV_64F);
				for (int i = 0; i < n; i++)
					for (int j = 0; j < d; j++)
						X.at<double>(i, j) = samples[i][j];
               
                int maxIter = 1000;
                double initMomentum = 0.5;
                double finalMomentum = 0.8;
                double eta = 500;
                double minGain = 0.01;

                cv::Mat Y(n, dims, CV_64F);
                cv::randn(Y, 0, 1);

                cv::Mat dY = cv::Mat::zeros(n, dims, CV_64F);
                cv::Mat iY = cv::Mat::zeros(n, dims, CV_64F);
                cv::Mat gains = cv::Mat::ones(n, dims, CV_64F);

				cv::Mat P = x2p(X, 1e-5, perplexity);
				cv::Mat tmp(P.size(), CV_64F);
				cv::transpose(P, tmp);
				P += tmp;
				P /= cv::sum(P);
				P *= 4;
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						P.at<double>(i, j) = std::max(1e-12, P.at<double>(i, j));

				for (int i = 0; i < maxIter; i++)
				{
					cv::pow(Y, 2, tmp);

					cv::Mat YSqr(n, 1, CV_64F);
					for (int i = 0; i < n; i++)
						YSqr.at<double>(i, 0) = cv::sum(tmp.row(i))[0];
				
					cv::Mat D(n, n, CV_64F);
					for (int j = 0; j < n; j++)
						for (int k = 0; k < n; k++)
							D.at<double>(j, k) = -2 * Y.row(j).dot(Y.col(k)) + 
								YSqr.at<double>(j, 0) + YSqr.at<double>(k, 0);

					D = 1 / (1 + D);
					for (int j = 0; j < n; j++)
						D.at<double>(j, j) = 0;

					cv::Mat Q = D / cv::sum(D);
					for (int j = 0; j < n; j++)
						for (int k = 0; k < n; k++)
							Q.at<double>(j, k) = std::max(1e-12, Q.at<double>(j, k));

					cv::Mat PQ = P - Q;
					tmp = cv::repeat(PQ.cols(i) * D.cols(i), dims, 1);
					tmp = tmp.t() * (Y.row(i) - Y);

					cv::Mat dY(n, 1, CV_64F);
					for (int j = 0; j < n; j++)
						dY.at<double>(j, 0) = cv::sum(tmp.cols(j));

					double momentum = i < 20 ? initMomentum : finalMomentum;
					gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + 
						(gains * 0.8) * ((dY > 0) == (iY == 0));
					for (int j = 0; j < n; j++)
						for (int k = 0; k < dims; k++)
							gains.at<double>(j, k) = std::max(minGain, gains.at<double>(j, k));

					iY = momentum * iY - eta * (gains * dY);
					Y = Y + iY;
					tmp.create(n, 1, CV_64F);
					for (int j = 0; j < n; j++)
						tmp.at<double>(j, 0) = cv::mean(Y.col(j));
					tmp = repeat(tmp, n, 1);
					Y = Y - tmp;

					if (i == 100)
						P /= 4;
				}

				return Y;
            }

        private:
			Tuple<double, cv::Mat> Hbeta(cv::Mat DistanceMatrix, double beta = 1.0)
			{
				cv::Mat P(DistanceMatrix.size(), CV_64F);
				cv::exp(-DistanceMatrix * beta, P);

				double sumP = cv::sum(P)[0];
				
				double H = std::log(sumP) + beta * sum(DistanceMatrix * P)[0] / sumP;
				P = P / sumP;

				return CreateTuple(H, P);
			}

            cv::Mat x2p(cv::Mat X, double tolerance = 1e-5, double perplexity = 30.0)
            {
                int n = X.rows, d = X.cols;

				cv::Mat tmp(X.size(), CV_64F);
				cv::pow(X, 2, tmp);

				cv::Mat XSqr(n, 1, CV_64F);
				for (int i = 0; i < n; i++)
					XSqr.at<double>(i, 0) = cv::sum(tmp.row(i))[0];

				cv::Mat D(n, n, CV_64F);
				for (int i = 0; i < n; i++)
					for (int j = 0; j < n; j++)
						D.at<double>(i, j) = -2 * X.row(i).dot(X.col(j)) + 
							XSqr.at<double>(i, 0) + XSqr.at<double>(j, 0);

				cv::Mat P = cv::Mat::zeros(n, n, CV_64F);
				cv::Mat beta = cv::Mat::ones(n, 1, CV_64F);
				double logU = std::log(perplexity);

				for (int i = 0; i < n; i++)
				{
					double INF = std::numeric_limits<double>::max();
					double betaMin = -INF;
					double betaMax = INF;

					cv::Mat Di(1, n - 1, CV_64F);
					for (int j = 0; j < i; j++)
						Di.at<double>(0, j) = D.at<double>(i, j);
					for (int j = i + 1; j < n; j++)
						Di.at<double>(0, j - 1) = D.at<double>(i, j);
				
					Tuple<double, cv::Mat> result = Hbeta(Di, beta.at<double>(i, 0));
					double H = result.Item1();
					cv::Mat Pi = result.Item2();

					double Hdiff = H - logU;
					int tries = 0;
					while (std::abs(Hdiff) > tolerance && tries < 50)
					{
						if (Hdiff > 0)
						{
							betaMin = beta.at<double>(i, 0);

							if (betaMax == INF || betaMax == -INF)
								beta.at<double>(i, 0) *= 2;
							else
								beta.at<double>(i, 0) = (beta.at<double>(i, 0) + betaMax) / 2;
						}
						else
						{
							betaMax = beta.at<double>(i, 0);

							if (betaMin == INF || betaMin == -INF)
								beta.at<double>(i, 0) = beta.at<double>(i, 0) / 2;
							else
								beta.at<double>(i, 0) = (beta.at<double>(i, 0) + betaMin) / 2;
						}

						result = Hbeta(Di, beta.at<double>(i, 0));
						H = result.Item1();
						Pi = result.Item2();

						Hdiff = H - logU;
						tries++;
					}

					for (int j = 0; j < i; j++)
						P.at<double>(i, j) = Di.at<double>(0, j);
					for (int j = i + 1; j < n; j++)
						P.at<double>(i, j) = Pi.at<double>(0, j - 1);
				}

				return P;
            }
        };
    }
}
}