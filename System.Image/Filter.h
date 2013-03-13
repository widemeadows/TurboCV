#pragma once

#include "../System/System.h"
#include "System.Image.h"
#include <cv.h>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        class Gradient
        {
        public:
            static Tuple<Mat, Mat> GetGradientKernel(double sigma, double epsilon);

            static Tuple<Mat, Mat> Gradient::GetGradient(const Mat& image, double sigma = 1.0);

            static vector<Mat> Gradient::GetOrientChannels(const Mat& sketchImage, int orientNum);
        };

        inline Tuple<Mat, Mat> Gradient::GetGradientKernel(double sigma, double epsilon)
        {
            int halfSize = (int)ceil(sigma * sqrt(-2 * log(sqrt(2 * CV_PI) * sigma * epsilon)));
            int size = halfSize * 2 + 1;
            double sum = 0, root;
            Mat dx(size, size, CV_64F), dy(size, size, CV_64F);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dx.at<double>(i, j) = Math::Gauss(i - halfSize, sigma) * Math::GaussDeriv(j - halfSize, sigma);
                    dy.at<double>(j, i) = dx.at<double>(i, j);
                    sum += dx.at<double>(i, j) * dx.at<double>(i, j);
                }
            }

            root = sqrt(sum);
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dx.at<double>(i, j) /= root;
                    dy.at<double>(i, j) /= root;
                }
            }

            return CreateTuple(dx, dy);
        }

        inline Tuple<Mat, Mat> Gradient::GetGradient(const Mat& image, double sigma)
        {
            Tuple<Mat, Mat> kernel = GetGradientKernel(sigma, 1e-2);
            Mat dxImage, dyImage;
            filter2D(image, dxImage, CV_64F, kernel.Item1());
            filter2D(image, dyImage, CV_64F, kernel.Item2());

            Mat orientImage(image.rows, image.cols, CV_64F);
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    double orient = atan2(dyImage.at<double>(i, j), dxImage.at<double>(i, j));
                    while (orient >= CV_PI)
                        orient -= CV_PI;
                    while (orient < 0)
                        orient += CV_PI;

                    orientImage.at<double>(i, j) = orient;
                }
            }

            Mat powerImage(image.rows, image.cols, CV_64F);
            for (int i = 0; i < image.rows; i++)
                for (int j = 0; j < image.cols; j++)
                    powerImage.at<double>(i, j) = sqrt(dyImage.at<double>(i, j) * dyImage.at<double>(i, j) +
                        dxImage.at<double>(i, j) * dxImage.at<double>(i, j));

            return CreateTuple(powerImage, orientImage);
        }

        inline vector<Mat> Gradient::GetOrientChannels(const Mat& sketchImage, int orientNum)
        {
            Tuple<Mat, Mat> gradient = GetGradient(sketchImage);
            Mat& powerImage = gradient.Item1();
            Mat& orientImage = gradient.Item2();
            int height = sketchImage.rows, width = sketchImage.cols;
            double orientBinSize = CV_PI / orientNum;

            vector<Mat> orientChannels;
            for (int i = 0; i < orientNum; i++)
                orientChannels.push_back(Mat::zeros(height, width, CV_64F));

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int o = (int)(orientImage.at<double>(i, j) / orientBinSize);
                    if (o < 0)
                        o = 0;
                    if (o >= orientNum)
                        o = orientNum - 1;

                    for (int k = -1; k <= 1; k++)
                    {
                        int newO = o + k;
                        double oRatio = 1 - abs((newO + 0.5) * orientBinSize - 
                            orientImage.at<double>(i, j)) / orientBinSize;
                        if (oRatio < 0)
                            oRatio = 0;
            
                        if (newO == -1)
                            newO = orientNum - 1;
                        if (newO == orientNum)
                            newO = 0;

                        orientChannels[newO].at<double>(i, j) += 
                            powerImage.at<double>(i, j) * oRatio;
                    }
                }
            }

            return orientChannels;
        }

        // http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
        inline Mat getLoGKernel(int ksize, double sigma, int ktype = CV_64F)
        {
            CV_Assert(ksize > 0 && ksize % 2 != 0);
            CV_Assert(ktype == CV_64F || ktype == CV_32F);

            int halfSize = ksize / 2;
            Mat kernel(ksize, ksize, ktype);

            double scale = -1 / (CV_PI * pow(sigma, 4));
            for (int i = 0; i < ksize; i++)
            {
                for (int j = i; j < ksize; j++)
                {
                    double y = i - halfSize, x = j - halfSize;
                    double tmp = -(x * x + y * y) / (2 * sigma * sigma);
                    double value = scale * (1 + tmp) * exp(tmp);

                    if (ktype == CV_64F)
                    {
                        kernel.at<double>(i, j) = value;
                        kernel.at<double>(j, i) = kernel.at<double>(i, j);
                    }
                    else
                    {
                        kernel.at<float>(i, j) = (float)value;
                        kernel.at<float>(j, i) = kernel.at<float>(i, j);
                    }
                }
            }

            return kernel;
        }

        inline vector<Mat> GetLoGPyramid(const Mat& image, const vector<double>& sigmas)
        {
            size_t sigmaNum = sigmas.size();
            vector<Mat> LoGPyramid(sigmaNum);

            for (int i = 0; i < sigmaNum; i++)
            {
                CV_Assert(sigmas[i] >= 0);

                int ksize = (int)(sigmas[i] * 6 + 1);
                if (ksize % 2 == 0)
                    ksize++;

                Mat kernel = getLoGKernel(ksize, sigmas[i], CV_64F);
                filter2D(image, LoGPyramid[i], CV_64F, kernel);
                LoGPyramid[i] = abs(LoGPyramid[i]) * pow(sigmas[i], 4); // pow(sigmas[i], 4) normalizes the integral
            }

            return LoGPyramid;
        }

        inline vector<Mat> GetDoGPyramid(const Mat& image, const vector<double>& sigmas)
        {
            size_t sigmaNum = sigmas.size();
            vector<Mat> GaussianPyramid(sigmaNum + 1);

            image.convertTo(GaussianPyramid[0], CV_64F);
            for (int i = 0; i < sigmaNum; i++)
            {
                CV_Assert(sigmas[i] >= 0);

                int ksize = (int)(sigmas[i] * 6 + 1);
                if (ksize % 2 == 0)
                    ksize++;

                Mat kernel = getGaussianKernel(ksize, sigmas[i], CV_64F);
                sepFilter2D(image, GaussianPyramid[i + 1], CV_64F, kernel, kernel);
            }

            vector<Mat> DoGPyramid;
            for (int i = 1; i <= sigmaNum; i++)
                DoGPyramid[i - 1] = GaussianPyramid[i] - GaussianPyramid[i - 1];

            return DoGPyramid;
        }
    }
}