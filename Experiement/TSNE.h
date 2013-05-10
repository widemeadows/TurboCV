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
            static cv::Mat Compute(ArrayList<ArrayList<T>> samples, int dims, 
                double perplexity = 30.0)
            {
                assert(samples.Count() > 0);

                int n = samples.Count();
                int d = samples[0].Count();

                cv::Mat tmp(n, d, CV_64F);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < d; j++)
                        tmp.at<double>(i, j) = samples[i][j];
                cv::Mat center(1, d, CV_64F);
                for (int j = 0; j < d; j++)
                    center.at<double>(0, j) = cv::mean(tmp.col(j))[0];
                for (int j = 0; j < n; j++)
                    tmp.row(j) -= center;

                printf("Perform PCA...\n");
                cv::PCA pca(tmp, cv::Mat(), CV_PCA_DATA_AS_ROW, 50);
                cv::Mat X = pca.project(tmp);

                int maxIter = 1000;
                double initMomentum = 0.5;
                double finalMomentum = 0.8;
                double eta = 500;
                double minGain = 0.01;

                cv::Mat Y(n, dims, CV_64F);
                cv::randn(Y, 0, 1);

                cv::Mat iY = cv::Mat::zeros(n, dims, CV_64F);
                cv::Mat gains = cv::Mat::ones(n, dims, CV_64F);

                printf("Compute P...\n");
                cv::Mat P = x2p(X, 1e-5, perplexity);
                P *= 4;
                P = cv::max(1e-12, P);
                for (int i = 0; i < maxIter; i++)
                {
                    cv::Mat D = cv::Mat::zeros(n, n, CV_64F);
                    for (int j = 0; j < n; j++)
                        for(int k = j + 1; k < n; k++)
                            D.at<double>(j, k) = D.at<double>(k, j) = rowDistance(Y, j, k);
                    D = 1 / (1 + D);
                    for (int j = 0; j < n; j++)
                        D.at<double>(j, j) = 0;

                    cv::Mat Q = D / cv::sum(D)[0];
                    Q = cv::max(1e-12, Q);

                    cv::Mat PQ = P - Q;
                    cv::Mat dY = cv::Mat::zeros(n, dims, CV_64F);
                    for (int j = 0; j < n; j++)
                        for (int k = 0; k < n; k++)
                            for (int m = 0; m < dims; m++)
                                dY.at<double>(j, m) += PQ.at<double>(j, k) * D.at<double>(j, k) *
                                    (Y.at<double>(j, m) - Y.at<double>(k, m));

                    cv::Mat tmp1;
                    ((cv::Mat)((dY > 0) != (iY > 0))).convertTo(tmp1, CV_64F);
                    tmp1 = cv::min(tmp1, 1.0);

                    cv::Mat tmp2;
                    ((cv::Mat)((dY > 0) == (iY > 0))).convertTo(tmp2, CV_64F);
                    tmp2 = cv::min(tmp2, 1.0);

                    gains = (gains + 0.2).mul(tmp1) + (gains * 0.8).mul(tmp2);
                    gains = cv::max(minGain, gains);

                    double momentum = i < 20 ? initMomentum : finalMomentum;
                    iY = momentum * iY - eta * gains.mul(dY);
                    Y = Y + iY;

                    cv::Mat center(1, dims, CV_64F);
                    for (int j = 0; j < dims; j++)
                        center.at<double>(0, j) = cv::mean(Y.col(j))[0];
                    for (int j = 0; j < n; j++)
                        Y.row(j) -= center;

                    if ((i + 1) % 10 == 0)
                    {
                        cv::Mat tmp;
                        cv::log(P / Q, tmp);
                        double C = cv::sum(P.mul(tmp))[0];
                        printf("Iteration %d: error is %f\n", i + 1, C);
                    }

                    if (i == 100)
                        P /= 4;
                }

                return Y;
            }

        private:
            static Tuple<double, cv::Mat> Hbeta(cv::Mat Di, double beta = 1.0)
            {
                cv::Mat gaussianD(Di.size(), CV_64F);
                cv::exp(Di * -beta, gaussianD);

                double sumGD = cv::sum(gaussianD)[0];
                
                double H = std::log(sumGD) + beta / sumGD * Di.dot(gaussianD);
                cv::Mat Pi = gaussianD / sumGD;

                return CreateTuple(H, Pi);
            }

            static inline double rowDistance(const cv::Mat& mat, int row1, int row2)
            {
                double sum = 0;

                for (int k = 0; k < mat.cols; k++)
                {
                    sum += (mat.at<double>(row1, k) - mat.at<double>(row2, k)) * 
                        (mat.at<double>(row1, k) - mat.at<double>(row2, k));
                }

                return sum;
            }

            static cv::Mat x2p(cv::Mat X, double tolerance = 1e-5, double perplexity = 30.0)
            {
                int n = X.rows, d = X.cols;

                cv::Mat D = cv::Mat::zeros(n, n, CV_64F);
                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++)
                        D.at<double>(i, j) = D.at<double>(j, i) = rowDistance(X, i, j);

                cv::Mat P = cv::Mat::zeros(n, n, CV_64F);
                cv::Mat beta = cv::Mat::ones(n, 1, CV_64F);
                double logU = std::log(perplexity);

                for (int i = 0; i < n; i++)
                {
                    double INF = 1e12;
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
                                beta.at<double>(i, 0) /= 2;
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
                        P.at<double>(i, j) = Pi.at<double>(0, j);
                    for (int j = i + 1; j < n; j++)
                        P.at<double>(i, j) = Pi.at<double>(0, j - 1);
                }

                P += P.t();
                return P / cv::sum(P)[0];
            }
        };
    }
}
}