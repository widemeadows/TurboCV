#pragma once

#include "../System/System.h"
#include <cv.h>
#include <highgui.h>
#include <numeric>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        const long long INF = numeric_limits<long long>::max();
        const double EPS = 1e-14;
        const int MAX_GRAYSCALE = 255;

        bool operator<(const Point& u, const Point& v)
        {
            if (u.x < v.x)
                return true;
            else if (u.x == v.x)
                return u.y < v.y;
            else
                return false;
        }

        struct PointHash
        {
            size_t operator()(const Point& v) const
            {
                return v.y * 10000000 + v.x;
            }
        };

        inline Mat reverse(Mat grayScaleImage) 
        {
            assert(grayScaleImage.type() == CV_8U);
            Mat result(grayScaleImage.size(), CV_8U);

            for (int i = 0; i < grayScaleImage.rows; i++)
                for (int j = 0; j < grayScaleImage.cols; j++)
                    result.at<uchar>(i, j) = MAX_GRAYSCALE - grayScaleImage.at<uchar>(i, j);

            return result;
        }

        template<typename RandomAccessIterator>
        inline void NormOneNormalize(const RandomAccessIterator& begin, const RandomAccessIterator& end)
        {
            double sum = 0;
            RandomAccessIterator curr = begin;

            do
            {
                sum += *curr;     
            } while (++curr != end);

            if (sum != 0)
            {
                curr = begin;

                do
                {
                    *curr /= sum;
                } while (++curr != end);
            }
        }

        template<typename RandomAccessIterator>
        inline void NormTwoNormalize(const RandomAccessIterator& begin, const RandomAccessIterator& end)
        {
            double sum = 0;
            RandomAccessIterator curr = begin;

            do
            {
                sum += (*curr) * (*curr);     
            } while (++curr != end);

            if (sum != 0)
            {
                double root = sqrt(sum);
                curr = begin;
                
                do
                {
                    *curr /= root;
                } while (++curr != end);
            }
        }

        inline size_t RoundIndex(int index, size_t period, bool cyclic = false)
        {
            if (cyclic)
            {
                while (index < 0)
                    index += period;

                return (size_t)index % period;
            }
            else
            {
                if (index < 0)
                    return 0;
                else if (index >= period)
                    return period - 1;
                else
                    return index;
            }
        }

        inline size_t FindBinIndex(double value, double minIncluded, double maxExcluded, 
            size_t intervalNum, bool cyclic = false)
        {
            assert(intervalNum > 0 && maxExcluded > minIncluded);

            double intervalSize = (maxExcluded - minIncluded) / intervalNum;

            int index = (int)(value / intervalSize);
            index = RoundIndex(index, intervalNum, cyclic);
            assert(index >= 0 && index < intervalNum);

            return index;
        }

        inline Mat FFTShift(const Mat& data)
        {
	        int width = data.cols, height = data.rows;
	        Mat result(data.rows, data.cols, data.type());

	        for (int i = 0; i < height / 2; i++)
	        {
		        for (int j = 0; j < width / 2; j++)
			        result.at<double>(i, j) = data.at<double>(i + (height + 1) / 2, j + (width + 1) / 2);

		        for (int j = 0; j < (width + 1) / 2; j++)
			        result.at<double>(i, j + width / 2) = data.at<double>(i + (height + 1) / 2, j);
	        }

	        for (int i = 0; i < (height + 1) / 2; i++)
	        {
		        for (int j = 0; j < width / 2; j++)
			        result.at<double>(i + height / 2, j) = data.at<double>(i, j + (width + 1) / 2);

		        for (int j = 0; j < (width + 1) / 2; j++)
			        result.at<double>(i + height / 2, j + width / 2) = data.at<double>(i, j);
	        }

	        return result;
        }

        inline vector<double> linspace(double start, double end, int pointNum)
        {
            double size = (end - start) / (pointNum - 1);

            vector<double> result;
            result.push_back(start);
            for (int i = 1; i < pointNum - 1; i++)
                result.push_back(result[i - 1] + size);
            result.push_back(end);

            assert(result.size() == pointNum);
            return result;
        }

        inline void imshow(const Mat& image, bool scale = true)
        {
            double maximum = 1e-14, minimum = 1e14;
            int type = image.type();
            Mat tmp(image.size(), CV_64F);

            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    double value;

                    if (type == CV_8U)
                        value = image.at<uchar>(i, j);
                    else if (type == CV_64F)
                        value = image.at<double>(i, j);
                    else if (type == CV_32S)
                        value = image.at<int>(i, j);
                    else if (type == CV_8S)
                        value = image.at<char>(i, j);
                    else if (type == CV_32F)
                        value = image.at<float>(i, j);

                    maximum = max(value, maximum);
                    minimum = min(value, minimum);
                    tmp.at<double>(i, j) = value;
                }
            }

            if (maximum > minimum)
            {
                for (int i = 0; i < tmp.rows; i++)
                    for (int j = 0; j < tmp.cols; j++)
                        tmp.at<double>(i, j) = (tmp.at<double>(i, j) - minimum) / 
                            (maximum - minimum);
            }

            imshow("OpenCV", tmp);
        }
    }
}