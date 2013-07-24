#pragma once

#include "../System/System.h"
#include <cassert>
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        //////////////////////////////////////////////////////////////////////////
        // Helper Functions
        //////////////////////////////////////////////////////////////////////////

        template<typename T>
        ArrayList<ArrayList<double>> NormalizeX(const ArrayList<ArrayList<T>>& X)
        {
            double maxSqrSum = 0;
            for (int i = 0; i < X.Count(); i++)
            {
                double sqrSum = 0;
                for (int j = 0; j < X[i].Count(); j++)
                    sqrSum += X[i][j] * X[i][j];

                if (sqrSum > maxSqrSum)
                    maxSqrSum = sqrSum;
            }

            double root = sqrt(maxSqrSum);
            if (root == 0)
                return X;
            else
            {
                ArrayList<ArrayList<double>> results(X.Count());
                for (int i = 0; i < X.Count(); i++)
                {
                    ArrayList<double> tmp(X[i].Count());
                    for (int j = 0; j < X[i].Count(); j++)
                        tmp[j] = X[i][j] / root;

                    results.Add(tmp);
                }

                return results;
            }
        }

        template<typename T>
        ArrayList<ArrayList<double>> PerformTSNE(const ArrayList<ArrayList<T>>& X, int yDim = 2)
        {
            cv::Mat Y = TSNE::Compute(NormalizeX(X), 30, yDim);

            ArrayList<ArrayList<double>> results(Y.rows);
            for (int i = 0; i < Y.rows; i++)
            {
                ArrayList<ArrayList<double>> tmp(Y.cols);
                for (int j = 0; j < Y.cols; j++)
                    tmp[j] = Y.at<double>(i, j);

                results.Add(tmp);
            }

            return results;
        }

        template<typename T>
        ArrayList<ArrayList<double>> PerformSNE(const ArrayList<ArrayList<T>>& X, int yDim = 2)
        {
            cv::Mat Y = SNE::Compute(NormalizeX(X), 30, yDim);

            ArrayList<ArrayList<double>> results(Y.rows);
            for (int i = 0; i < Y.rows; i++)
            {
                ArrayList<ArrayList<double>> tmp(Y.cols);
                for (int j = 0; j < Y.cols; j++)
                    tmp[j] = Y.at<double>(i, j);

                results.Add(tmp);
            }

            return results;
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for TSNE
        //////////////////////////////////////////////////////////////////////////

        class TSNE
        {
        public:
            template<typename T>
            static cv::Mat Compute(ArrayList<ArrayList<T>> samples, double perplexity = 30.0, int yDim = 2)
            {
                std::assert(samples.Count() > 0);
                int n = samples.Count(), xDims = samples[0].Count();

                cv::Mat X(n, xDims, CV_64F);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < xDims; j++)
                        X.at<double>(i, j) = samples[i][j];

                return Compute(X, perplexity, yDim);
            }

            static cv::Mat Compute(cv::InputArray samples, double perplexity = 30.0, int yDim = 2);

        private:
            static cv::Mat getDistanceMatrix(cv::InputArray matrix);
            static cv::Mat normalizeVectors(cv::InputOutputArray matrix);
            static cv::Mat getP(cv::Mat X, double tolerance = 1e-5, double perplexity = 30.0);
        };


        //////////////////////////////////////////////////////////////////////////
        // APIs for SNE
        //////////////////////////////////////////////////////////////////////////

        class SNE
        {
        public:
            template<typename T>
            static cv::Mat Compute(ArrayList<ArrayList<T>> samples, double perplexity = 30.0, int yDim = 2)
            {
                assert(samples.Count() > 0);
                int n = samples.Count(), xDims = samples[0].Count();

                cv::Mat X(n, xDims, CV_64F);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < xDims; j++)
                        X.at<double>(i, j) = samples[i][j];

                return Compute(X, perplexity, yDim);
            }

            static cv::Mat Compute(cv::InputArray samples, double perplexity = 30.0, int yDim = 2);

        private:
            static cv::Mat getDistanceMatrix(cv::InputArray matrix);
            static cv::Mat normalizeVectors(cv::InputOutputArray matrix);
            static cv::Mat getP(cv::Mat X, double tolerance = 1e-5, double perplexity = 30.0);
        };
    }
}
}