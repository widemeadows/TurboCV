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
        Mat calcProb(const Mat& x, const GMMFeature& model)
        {
            int N = x.rows;
            int D = x.cols;
            int K = model.miu.Count();

            Mat prob = Mat::zeros(N, K, CV_64F);
            for (int i = 0; i < K; i++)
            {
                Mat invSigma = model.sigma[i].inv();
                Mat dif = x - repeat(model.miu[i], N, 1);
                Mat gauss;
                reduce((dif * invSigma).mul(dif, -0.5), gauss, 1, CV_REDUCE_SUM);
                exp(gauss, gauss);
                gauss *= model.a[i] * pow(2 * CV_PI, -D / 2.0) * sqrt(determinant(invSigma));

                gauss.copyTo(prob.col(i));
            }

            return prob;
        }

        double GetDistance(const GMMFeature& u, const GMMFeature& v)
        {
            int D = u.miu[0].cols;
            int K = u.a.Count();
            ArrayList<double> a1, a2, a12;
            ArrayList<Mat> miu1, miu2, miu12;
            ArrayList<Mat> sigma1, sigma2, sigma12;

            for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
            {
                a1.Add(u.a[i] * u.a[j]);
                a2.Add(v.a[i] * v.a[j]);
                a12.Add(u.a[i] * v.a[j]);

                miu1.Add(u.miu[i] - u.miu[j]);
                miu2.Add(v.miu[i] - v.miu[j]);
                miu12.Add(u.miu[i] - v.miu[j]);

                sigma1.Add(u.sigma[i] + u.sigma[j]);
                sigma2.Add(v.sigma[i] + v.sigma[j]);
                sigma12.Add(u.sigma[i] + v.sigma[j]);
            }

            Mat origin = Mat::zeros(1, D, CV_64F);
            Mat prob1 = calcProb(origin, GMMFeature(a1, miu1, sigma1));
            Mat prob2 = calcProb(origin, GMMFeature(a2, miu2, sigma2));
            Mat prob12 = calcProb(origin, GMMFeature(a12, miu12, sigma12));

            double d1 = sum(prob1)[0];
            double d2 = sum(prob2)[0];
            double d12 = sum(prob12)[0] * 2.0;

            return (d1 + d2 - d12) / sqrt(1.0 + 4 * d2);
        }

        //////////////////////////////////////////////////////////////////////////
        // Gaussian Mixture Model
        //////////////////////////////////////////////////////////////////////////

        Mat regularCov(const Mat& cov)
        {
            return cov + Mat::eye(cov.size(), cov.type()) * 0.1;
        }

        GMMFeature initParams(const Mat& x, int K)
        {   
            int N = x.rows;
            int D = x.cols;
         
            // 随机选择K个向量作为聚类中心
            ArrayList<Mat> miu(K);
            auto idx = RandomPermutate(N, K);
            for (int i = 0; i < K; i++)
                miu[i] = x.row(idx[i]);

            // 将向量分散到K个聚类中
            ArrayList<int> clusterId(N), clusterSize(K);
            for (int i = 0; i < N; i++)
            {
                double minDis = numeric_limits<double>::max();
                int minIdx = -1;

                for (int j = 0; j < K; j++)
                {
                    auto dif = x.row(i) - miu[j];
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
                    x.row(j).copyTo(Xk.row(--clusterSize[i]));

                Mat cov, mean;
                calcCovarMatrix(Xk, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
                sigma[i] = regularCov(cov / (Xk.rows - 1));
            }

            return GMMFeature(p, miu, sigma);
        }

        Group<Mat, GMMFeature> GMM(const Mat& x, int K)
        {
            return GMM(x, K, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1e-6));
        }

        Group<Mat, GMMFeature> GMM(const Mat& x, int K, const TermCriteria& termCriteria)
        {
            int N = x.rows;
            int D = x.cols;

            int maxIter = 100;
            double epsilon = 1e-6;
            if (termCriteria.type & CV_TERMCRIT_ITER)
                maxIter = termCriteria.maxCount;
            if (termCriteria.type & CV_TERMCRIT_EPS)
                epsilon = termCriteria.epsilon;

            double prevLoss = -epsilon;

            auto model = initParams(x, K);;
            Mat prob;

            // EM算法
            for (int iter = 0; iter < maxIter; iter++)
            {
                // Estimation Step
                prob = calcProb(x, model);

                Mat probSum;
                reduce(prob, probSum, 1, CV_REDUCE_SUM);
                Mat gamma = prob / repeat(probSum, 1, K);

                // Maximization Step
                for (int i = 0; i < K; i++)
                {
                    double Nk = sum(gamma.col(i))[0];
                    model.a[i] = Nk / N;
                    model.miu[i] = 1.0 / Nk * (gamma.col(i).t() * x);

                    Mat dif1 = x - repeat(model.miu[i], N, 1);
                    Mat dif2 = dif1.mul(repeat(gamma.col(i), 1, D));
                    model.sigma[i] = regularCov(dif1.t() * dif2 / Nk);
                }

                log(probSum, probSum);
                double loss = sum(probSum)[0];
                if (abs(loss - prevLoss) < epsilon)
                    break;
                prevLoss = loss;
            }

            return CreateGroup(prob, model);
        }
    }
}
}