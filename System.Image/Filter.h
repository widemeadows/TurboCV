#pragma once

#include "../System/System.h"
#include "System.Image.h"
#include <cv.h>
#include <tuple>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        class Gradient
        {
        public:
            static tuple<Mat, Mat> GetGradientKernel(double sigma, double epsilon);

            static tuple<Mat, Mat> Gradient::GetGradient(const Mat& image);

            static vector<Mat> Gradient::GetOrientChannels(const Mat& sketchImage, int orientNum);
        };

        inline tuple<Mat, Mat> Gradient::GetGradientKernel(double sigma, double epsilon)
        {
            double halfSize = ceil(sigma * sqrt(-2 * log(sqrt(2 * CV_PI) * sigma * epsilon)));
            double size = halfSize * 2 + 1;
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

            return make_tuple(dx, dy);
        }

        inline tuple<Mat, Mat> Gradient::GetGradient(const Mat& image)
        {
            tuple<Mat, Mat> kernel = GetGradientKernel(1.0, 1e-2);
            Mat dxImage, dyImage;
            filter2D(image, dxImage, CV_64F, get<0>(kernel));
            filter2D(image, dyImage, CV_64F, get<1>(kernel));

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

            return make_tuple(powerImage, orientImage);
        }

        inline vector<Mat> Gradient::GetOrientChannels(const Mat& sketchImage, int orientNum)
        {
            tuple<Mat, Mat> gradient = GetGradient(sketchImage);
            Mat& powerImage = get<0>(gradient);
            Mat& orientImage = get<1>(gradient);
            int height = sketchImage.rows, width = sketchImage.cols;
            double orientBinSize = CV_PI / orientNum;

            vector<Mat> orientChannels;
            for (int i = 0; i < orientNum; i++)
                orientChannels.push_back(Mat::zeros(height, width, CV_64F));

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int o = orientImage.at<double>(i, j) / orientBinSize;
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
            int sigmaNum = sigmas.size();
            vector<Mat> levels(sigmaNum);

            for (int i = 0; i < sigmaNum; i++)
            {
                CV_Assert(sigmas[i] >= 0);

                int ksize = sigmas[i] * 6 + 1;
                if (ksize % 2 == 0)
                    ksize++;

                Mat kernel = getLoGKernel(ksize, sigmas[i], CV_64F);
                filter2D(image, levels[i], CV_64F, kernel);
                levels[i] = (sigmas[i] * sigmas[i]) * abs(levels[i]);
            }

            return levels;
        }

        inline vector<tuple<Mat, double>> GetDOGPyramid(const Mat& base, double sigmaInit, 
            double sigmaStep, int levels)
        {
            Mat* GaussianPyramid = new Mat[levels + 3];
            double* sigmas = new double[levels + 3];

            sigmas[0] = 0;
            sigmas[1] = sigmaInit;
            for (int i = 2; i < levels + 3; i++)
                sigmas[i] = sigmas[i - 1] * sigmaStep;

            base.convertTo(GaussianPyramid[0], CV_64F);
            for (int i = 1; i < levels + 3; i++)
            {
                int kernelSize = sigmas[i] * 6;
                if (kernelSize % 2 == 0)
                    kernelSize++;

                Mat kernel = getGaussianKernel(kernelSize, sigmas[i], CV_64F);
                sepFilter2D(base, GaussianPyramid[i], CV_64F, kernel, kernel);
            }

            vector<tuple<Mat, double>> result;
            for (int i = 1; i < levels + 3; i++)
            {
                Mat diff;
                absdiff(GaussianPyramid[i], GaussianPyramid[i - 1], diff);
                result.push_back(make_tuple(diff, sigmas[i] * 3));
            }

            //for (int i = 0; i < levels + 2; i++)
            //{
            //    double maxi = -1e14, mini = 1e14;

            //    for (int j = 0; j < GaussianPyramid[i].rows; j++)
            //        for (int k = 0; k < GaussianPyramid[i].cols; k++)
            //        {
            //            if (abs(get<0>(result[i]).at<double>(j, k)) > maxi)
            //                maxi = abs(get<0>(result[i]).at<double>(j, k));
            //            if (abs(get<0>(result[i]).at<double>(j, k)) < mini)
            //                mini = abs(get<0>(result[i]).at<double>(j, k));
            //        }

            //    Mat tmp(get<0>(result[i]).rows, get<0>(result[i]).cols, CV_64F);
            //    for (int j = 0; j < tmp.rows; j++)
            //        for (int k = 0; k < tmp.cols; k++)
            //        {
            //            tmp.at<double>(j, k) = (abs(get<0>(result[i]).at<double>(j, k)) - mini) / (maxi - mini);
            //        }

            //    imshow("win", tmp);
            //    waitKey(0);
            //}

            delete[] sigmas;
            delete[] GaussianPyramid;
            return result;
        }
    }
}