#pragma once

#include <Eigen/Eigen>
#include <cv.h>
#include <algorithm>
#include "../System/System.h"

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
        void eigen2cv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, 
                      cv::Mat& dst)
        {
            if (!(src.Flags & Eigen::RowMajorBit))
            {
                cv::Mat _src(src.cols(), src.rows(), cv::DataType<_Tp>::type,
                    (void*)src.data(), src.stride() * sizeof(_Tp));
                cv::transpose(_src, dst);
            }
            else
            {
                cv::Mat _src(src.rows(), src.cols(), cv::DataType<_Tp>::type,
                    (void*)src.data(), src.stride() * sizeof(_Tp));
                _src.copyTo(dst);
            }
        }

        template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
        void cv2eigen(const cv::Mat& src, 
                      Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst)
        {
            CV_DbgAssert(src.rows == _rows && src.cols == _cols);

            if (!(dst.Flags & Eigen::RowMajorBit))
            {
                cv::Mat _dst(src.cols, src.rows, cv::DataType<_Tp>::type,
                    dst.data(), (size_t)(dst.stride() * sizeof(_Tp)));
                if (src.type() == _dst.type())
                    cv::transpose(src, _dst);
                else if (src.cols == src.rows)
                {
                    src.convertTo(_dst, _dst.type());
                    cv::transpose(_dst, _dst);
                }
                else
                    cv::Mat(src.t()).convertTo(_dst, _dst.type());
                CV_DbgAssert(_dst.data == (uchar*)dst.data());
            }
            else
            {
                cv::Mat _dst(src.rows, src.cols, cv::DataType<_Tp>::type,
                    dst.data(), (size_t)(dst.stride() * sizeof(_Tp)));
                src.convertTo(_dst, _dst.type());
                CV_DbgAssert(_dst.data == (uchar*)dst.data());
            }
        }

        void eigen(cv::InputArray A, cv::InputArray B, 
                   cv::OutputArray values, cv::OutputArray vectors)
        {
            cv::Mat mat = A.getMat().inv() * B.getMat();

            Eigen::MatrixXd tmp(mat.rows, mat.cols);
            cv2eigen(mat, tmp);

            Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(tmp);

            cv::Mat eigenVectors, eigenValues;
            eigen2cv(Eigen::MatrixXd(eigenSolver.eigenvalues().real()), eigenVectors);
            eigen2cv(Eigen::MatrixXd(eigenSolver.eigenvalues().real()), eigenValues);

            ArrayList<Group<double, int>> valueAndIndexes;
            for (int i = 0; i < eigenValues.rows; i++)
                valueAndIndexes.Add(CreateGroup(eigenValues.at<double>(i, 0), i));

            std::sort(valueAndIndexes.begin(), valueAndIndexes.end());

            vectors.create(eigenVectors.size(), CV_64F);
            values.create(eigenValues.size(), CV_64F);
            cv::Mat sortedVectors = vectors.getMat(), sortedValues = values.getMat();
            for (int i = 0; i < eigenValues.rows; i++)
            {
                int index = valueAndIndexes[i].Item2();
                eigenValues.row(index).copyTo(sortedValues.row(i));
                eigenVectors.row(index).copyTo(sortedVectors.row(i));
            }
        }
    }
}
}