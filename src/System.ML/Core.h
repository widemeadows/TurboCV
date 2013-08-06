#pragma once

#include "../System/System.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        inline cv::Mat diag(cv::Mat src)
        {
            cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

            for (int i = 0; i < src.rows; i++)
                dst.at<double>(i, i) = src.at<double>(i, i);

            return dst;
        }

        inline cv::Mat mean(cv::Mat src, bool dataAsRow = true)
        {
            cv::Mat tmp = src;
            if (!dataAsRow)
                tmp = src.t();

            cv::Mat avg = cv::Mat::zeros(1, tmp.cols, CV_64F);
            for (int i = 0; i < tmp.rows; i++)
                for (int j = 0; j < tmp.cols; j++)
                    avg.at<double>(0, j) += tmp.at<double>(i, j);

            for (int i = 0; i < tmp.cols; i++)
                avg.at<double>(0, i) /= tmp.rows;

            if (dataAsRow)
                return avg;
            else
                return avg.t();
        }

        template<typename T>
        ArrayList<ArrayList<T>> PCAWhiten(const ArrayList<ArrayList<T>>& vecs)
        {
            if (vecs.Count() == 0)
                return vecs;

            int nVec = vecs.Count(), nDim = vecs[0].Count();

            cv::Mat x(nDim, nVec, CV_64F);
            for (int i = 0; i < nVec; i++)
                for (int j = 0; j < nDim; j++)
                    x.at<double>(j, i) = vecs[i][j];

            cv::Mat avg = mean(x, false);
            x -= cv::repeat(avg, 1, x.cols);

            cv::Mat sigma = x * x.t() / nVec;

            cv::Mat eigenVectors, eigenValues;
            eigen(sigma, eigenValues, eigenVectors);

            cv::Mat xPCA = eigenVectors * x;

            ArrayList<ArrayList<T>> result;
            for (int i = 0; i < nVec; i++)
            {
                ArrayList<T> vec;
                for (int j = 0; j < nDim; j++)
                    vec.Add(xPCA.at<double>(j, i));

                result.Add(vec);
            }
            return result;
        }

        double Kmeans(cv::InputArray _data, int K, cv::InputOutputArray _bestLabels,
            cv::TermCriteria criteria, int attempts, int flags, cv::OutputArray _centers);
    }
}
}