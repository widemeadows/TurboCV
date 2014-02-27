#include "../System/System.h"
#include <numeric>
#include <cv.h>
#include "Clustering.h"
using namespace TurboCV::System;
using namespace std;
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        Group<ArrayList<double>, ArrayList<Mat>, ArrayList<Mat>> 
        initParams(const Mat& X, int K)
        {   
            int N = X.rows;
            int D = X.cols;
         
            // 随机选择K个向量作为聚类中心
            ArrayList<Mat> u(K);
            auto idx = RandomPermutate(N, K);
            for (int i = 0; i < K; i++)
                u[i] = X.row(idx[i]);

            // 将向量分散到K个聚类中
            ArrayList<int> clusterId(N), clusterSize(K);
            for (int i = 0; i < N; i++)
            {
                double minDis = numeric_limits<double>::max();
                int minIdx = -1;

                for (int j = 0; j < K; j++)
                {
                    auto dif = X.row(i) - u[j];
                    double sqrDis = dif.dot(dif);

                    if (sqrDis < minDis)
                    {
                        minDis = sqrDis;
                        minIdx = j;
                    }
                }

                clusterId[i] = minIdx;
                clusterSize[minIdx]++;
            }

            // 计算每个聚类的协方差和选择概率
            ArrayList<double> p(K);
            ArrayList<Mat> sigma(K);
            for (int i = 0; i < K; i++)
            {
                p[i] = (double)clusterSize[i] / (double)N;

                Mat Xk(clusterSize[i], D, CV_64F);
                for (int j = 0; j < N; j++)
                if (clusterId[j] == i)
                    X.row(j).copyTo(Xk.row(--clusterSize[i]));

                Mat cov, mean;
                calcCovarMatrix(Xk, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
                sigma[i] = cov / (Xk.rows - 1);
            }

            return CreateGroup(p, u, sigma);
        }

        Mat calcProb(const Mat& X, const ArrayList<double>& p, 
            const ArrayList<Mat>& u, const ArrayList<Mat>& sigma)
        {
            int N = X.rows;
            int D = X.cols;
            int K = u.Count();

            Mat prob = Mat::zeros(N, K, CV_64F);
            for (int i = 0; i < K; i++)
            {
                Mat invSigma = sigma[i].inv();
                double coef = p[i] * pow(2 * CV_PI, -D / 2.0) * sqrt(determinant(invSigma));

                for (int j = 0; j < N; j++)
                {
                    Mat dif = X.row(j) - u[i];
                    prob.at<double>(j, i) = coef * exp(-0.5 * 
                        Mat(dif * invSigma * dif.t()).at<double>(0, 0));
                }
            }

            for (int i = 0; i < N; i++)
            {
                double probSum = sum(prob.row(i))[0];

                for (int j = 0; j < K; j++)
                    prob.at<double>(i, j) /= probSum;
            }

            return prob;
        }

        Group<Mat, Group<ArrayList<double>, ArrayList<Mat>, ArrayList<Mat>>> 
        GMM(const Mat& X, int K)
        {
            int N = X.rows;
            int D = X.cols;

            auto result = initParams(X, K);
            ArrayList<double> p = result.Item1();
            ArrayList<Mat> u = result.Item2();
            ArrayList<Mat> sigma = result.Item3();
            Mat prob;

            // EM算法
            for (int iter = 0; iter < 100; iter++)
            {
                // Estimation Step
                prob = calcProb(X, p, u, sigma);

                // Maximization Step
                for (int i = 0; i < K; i++)
                {
                    double Nk = sum(prob.col(i))[0];
                    p[i] = Nk / N;
                    u[i] = 1.0 / Nk * (prob.col(i).t() * X);

                    sigma[i] = Mat::zeros(D, D, CV_64F);
                    for (int j = 0; j < N; j++)
                    {
                        Mat dif = X.row(j) - u[i];
                        sigma[i] += dif.t() * dif * prob.at<double>(j, i);
                    }
                    sigma[i] /= Nk;
                }
            }

            return CreateGroup(prob, CreateGroup(p, u, sigma));
        }
    }
}
}