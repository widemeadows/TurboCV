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
        inline tuple<Mat, Mat> GetGradientKernel(double sigma, double epsilon)
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

        inline tuple<Mat, Mat> GetGradient(const Mat& image)
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

        inline vector<Mat> GetOrientChannels(const Mat& sketchImage, int orientNum)
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
    }
}