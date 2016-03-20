#include "Reduction.h"
#include <cv.h>
#include <cstdio>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        //////////////////////////////////////////////////////////////////////////
        // TSNE
        //////////////////////////////////////////////////////////////////////////

        Mat TSNE::Compute(InputArray samples, double perplexity, int yDim)
        {
            Mat X = samples.getMat().clone();
            normalizeVectors(X);

            const int n = X.rows;
            const double eps = 1e-12;
            const int MAX_ITER = 1000;
            const double INIT_MOMENTUM = 0.5;
            const double FINAL_MOMENTUM = 0.8;
            const double eta = 500;
            const double MIN_GAIN = 0.01;

            Mat Y(n, yDim, CV_64F);
            randn(Y, 0, 1);

            Mat iY = Mat::zeros(n, yDim, CV_64F);
            Mat gains = Mat::ones(n, yDim, CV_64F);

            printf("Compute P...\n");
            Mat P = getP(X, 1e-5, perplexity);
            P *= 4;
            P = max(eps, P);

            for (int i = 0; i < MAX_ITER; i++)
            {
                Mat D = getDistanceMatrix(Y);
                D = 1 / (1 + D);
                for (int j = 0; j < n; j++)
                    D.at<double>(j, j) = 0;

                Mat Q = D / sum(D)[0];
                Q = max(eps, Q);

                Mat PQ = P - Q;
                Mat dY = Mat::zeros(n, yDim, CV_64F);
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        double tmp = PQ.at<double>(j, k) * D.at<double>(j, k);

                        for (int m = 0; m < yDim; m++)
                            dY.at<double>(j, m) += tmp *
                            (Y.at<double>(j, m) - Y.at<double>(k, m));
                    }
                }

                Mat tmp1;
                ((Mat)((dY > 0) != (iY > 0))).convertTo(tmp1, CV_64F);
                tmp1 = min(tmp1, 1.0);

                Mat tmp2;
                ((Mat)((dY > 0) == (iY > 0))).convertTo(tmp2, CV_64F);
                tmp2 = min(tmp2, 1.0);

                gains = (gains + 0.2).mul(tmp1) + (gains * 0.8).mul(tmp2);
                gains = max(MIN_GAIN, gains);

                double momentum = i < 20 ? INIT_MOMENTUM : FINAL_MOMENTUM;
                iY = momentum * iY - eta * gains.mul(dY);

                Y = Y + iY;
                normalizeVectors(Y);

                if ((i + 1) % 10 == 0)
                {
                    Mat tmp;
                    log(P / Q, tmp);
                    double C = sum(P.mul(tmp))[0];
                    printf("Iteration %d: error is %f\n", i + 1, C);
                }

                if (i == 100)
                    P /= 4;
            }

            return Y;
        }

        Mat TSNE::getDistanceMatrix(InputArray matrix)
        {
            Mat mat = matrix.getMat();
            int n = mat.rows, d = mat.cols;
            Mat D = Mat::zeros(n, n, CV_64F);

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

        Mat TSNE::normalizeVectors(InputOutputArray matrix)
        {
            Mat mat = matrix.getMat();

            Mat center = Mat::zeros(1, mat.cols, CV_64F);
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

        Mat TSNE::getP(Mat X, double tolerance, double perplexity)
        {
            int n = X.rows, d = X.cols;
            Mat D = getDistanceMatrix(X);
            Mat P = Mat::zeros(n, n, CV_64F);
            ArrayList<double> sigmaSqr(n, 1.0);
            const double logU = std::log(perplexity);
            const double INF = 1e12;

            #pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                double lowerBound = -INF;
                double upperBound = INF;

                Mat Di(1, n - 1, CV_64F);
                for (int j = 0; j < i; j++)
                    Di.at<double>(0, j) = D.at<double>(i, j);
                for (int j = i + 1; j < n; j++)
                    Di.at<double>(0, j - 1) = D.at<double>(i, j);

                Mat Pi;
                double Hdiff = 0;
                int tries = 0;
                do 
                {
                    Mat gaussDi(Di.size(), CV_64F);
                    exp(Di / -sigmaSqr[i], gaussDi);
                    double sumGDi = sum(gaussDi)[0];

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

            P += P.t();
            P /= sum(P)[0];
            return P;
        }


        //////////////////////////////////////////////////////////////////////////
        // SNE
        //////////////////////////////////////////////////////////////////////////

        Mat SNE::Compute(InputArray samples, double perplexity, int yDim)
        {
            Mat X = samples.getMat().clone();
            normalizeVectors(X);

            const int n = X.rows;
            const double eps = 1e-12;
            const int MAX_ITER = 1000;
            const double INIT_MOMENTUM = 0.5;
            const double FINAL_MOMENTUM = 0.8;
            const double eta = 0.05;
            const double jitterDecay = 0.99;
            double jitter = 0.3;

            Mat Y(n, yDim, CV_64F);
            randn(Y, 0, 1);

            Mat iY = Mat::zeros(n, yDim, CV_64F);

            printf("Compute P...\n");
            Mat P = getP(X, 1e-5, perplexity);
            P = max(eps, P);

            for (int i = 0; i < MAX_ITER; i++)
            {
                Mat D = getDistanceMatrix(Y);
                Mat gaussD;
                exp(-D, gaussD);
                for (int j = 0; j < n; j++)
                    gaussD.at<double>(j, j) = 0;

                Mat Q(gaussD.size(), CV_64F);
                for (int j = 0; j < Q.rows; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < Q.cols; k++)
                        sum += gaussD.at<double>(j, k);

                    for (int k = 0; k < Q.cols; k++)
                        Q.at<double>(j, k) = gaussD.at<double>(j, k) / sum;
                }
                Q = max(eps, Q);

                Mat PQ = P - Q;
                Mat dY = Mat::zeros(n, yDim, CV_64F);
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        double tmp = PQ.at<double>(j, k) + PQ.at<double>(k, j);

                        for (int m = 0; m < yDim; m++)
                            dY.at<double>(j, m) += tmp *
                            (Y.at<double>(j, m) - Y.at<double>(k, m));
                    }
                }

                double momentum = i < 750? INIT_MOMENTUM : FINAL_MOMENTUM;
                iY = momentum * iY - eta * dY;

                Y += iY;

                Mat normalDistribution(Y.size(), CV_64F);
                randn(normalDistribution, 0, 1);
                Y += jitter * normalDistribution;
                jitter *= jitterDecay;

                normalizeVectors(Y);

                if ((i + 1) % 10 == 0)
                {
                    Mat tmp;
                    log(P / Q, tmp);
                    double C = sum(P.mul(tmp))[0];
                    printf("Iteration %d: error is %f\n", i + 1, C);
                }
            }

            return Y;
        }

        Mat SNE::getDistanceMatrix(InputArray matrix)
        {
            Mat mat = matrix.getMat();
            int n = mat.rows, d = mat.cols;
            Mat D = Mat::zeros(n, n, CV_64F);

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

        Mat SNE::normalizeVectors(InputOutputArray matrix)
        {
            Mat mat = matrix.getMat();

            Mat center = Mat::zeros(1, mat.cols, CV_64F);
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

        Mat SNE::getP(Mat X, double tolerance, double perplexity)
        {
            int n = X.rows, d = X.cols;
            Mat D = getDistanceMatrix(X);
            Mat P = Mat::zeros(n, n, CV_64F);
            ArrayList<double> sigmaSqr(n, 1.0);
            const double logU = std::log(perplexity);
            const double INF = 1e12;

            #pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                double lowerBound = -INF;
                double upperBound = INF;

                Mat Di(1, n - 1, CV_64F);
                for (int j = 0; j < i; j++)
                    Di.at<double>(0, j) = D.at<double>(i, j);
                for (int j = i + 1; j < n; j++)
                    Di.at<double>(0, j - 1) = D.at<double>(i, j);

                Mat Pi;
                double Hdiff = 0;
                int tries = 0;
                do 
                {
                    Mat gaussDi(Di.size(), CV_64F);
                    exp(Di / -sigmaSqr[i], gaussDi);
                    double sumGDi = sum(gaussDi)[0];

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
    }
}
}