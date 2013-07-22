#pragma once

#include "../System/System.h"
#include "System.Image.h"
#include "Filter.h"
#include <cv.h>
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        ///////////////////////////////////////////////////////////////////////

        class Test : public LocalFeature
        {
        public:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage);

            virtual String GetName() const { return "test@1500"; };

        private:
            static Descriptor GetDescriptor1(const vector<Mat>& filteredOrientImages, 
                const Point& center, int blockSize, int cellNum);

            static Descriptor GetDescriptor2(const vector<Mat>& enlargeChannels, 
                const vector<Mat>& enlargeFilteredChannels,
                const Point& center, const Mat& weights, int blockSize, int cellNum);

            vector<Mat> GetChannels(const Mat& sketchImage, int orientNum)
            {
                int sigma = 4, lambda = 10, ksize = sigma * 6 + 1;
                vector<Mat> channels(orientNum);

                for (int i = 0; i < orientNum; i++)
                {
                    Mat kernel = getGaborKernel(Size(ksize, ksize), sigma, 
                        CV_PI / orientNum * i, lambda, 1, 0);

                    filter2D(sketchImage, channels[i], CV_64F, kernel);
                    channels[i] = abs(channels[i]);
                }

                return channels;
            }
        };

        inline LocalFeatureVec Test::GetFeature(const Mat& sketchImage)
        {
            int orientNum = 9, sampleNum = 28, blockSize = 92, cellNum = 4;

            int cellSize = blockSize / cellNum, kernelSize = cellSize * 2 + 1;
            Mat tentKernel(kernelSize, kernelSize, CV_64F);
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    double ratio = 1 - sqrt((i - cellSize) * (i - cellSize) + 
                        (j - cellSize) * (j - cellSize)) / cellSize;
                    if (ratio < 0)
                        ratio = 0;

                    tentKernel.at<double>(i, j) = ratio;
                }
            }
            //normalize(tentKernel, tentKernel, 1, 0, NORM_L1);

            vector<Mat> orientChannels = GetChannels(sketchImage, orientNum);
            vector<Mat> filteredOrientChannels(orientNum);
            for (int i = 0; i < orientNum; i++)
                filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

            LocalFeatureVec feature;
            vector<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor descriptor = GetDescriptor1(filteredOrientChannels, center, 
                    blockSize, cellNum);
                feature.Add(descriptor);
            }

            //vector<Mat> enlargeChannels(orientNum);
            //for (int i = 0; i < orientNum; i++)
            //    copyMakeBorder(orientChannels[i], enlargeChannels[i], cellSize, cellSize, 
            //        cellSize, cellSize, BORDER_DEFAULT | BORDER_ISOLATED);

            //vector<Mat> enlargeFilteredChannels(orientNum);
            //for (int i = 0; i < orientNum; i++)
            //    copyMakeBorder(filteredOrientChannels[i], enlargeFilteredChannels[i], cellSize, cellSize, 
            //        cellSize, cellSize, BORDER_DEFAULT | BORDER_ISOLATED);

            //for (Point center : centers)
            //{
            //    Descriptor descriptor = GetDescriptor2(enlargeChannels, enlargeFilteredChannels,
            //        center, tentKernel, blockSize, cellNum);
            //    feature.push_back(descriptor);
            //}

            return feature;
        }

        inline Descriptor Test::GetDescriptor1(const vector<Mat>& filteredOrientChannels, 
            const Point& center, int blockSize, int cellNum)
        {
            int height = filteredOrientChannels[0].rows, 
                width = filteredOrientChannels[0].cols;
            int cellSize = blockSize / cellNum;
            int expectedTop = center.y - blockSize / 2,
                expectedLeft = center.x - blockSize / 2,
                orientNum = filteredOrientChannels.size();
            int dims[] = { cellNum, cellNum, orientNum };
            Mat hist(3, dims, CV_64F);

            for (int i = 0; i < cellNum; i++)
            {
                for (int j = 0; j < cellNum; j++)
                {
                    for (int k = 0; k < orientNum; k++)
                    {
                        int r = (int)(expectedTop + (i + 0.5) * cellSize),
                            c = (int)(expectedLeft + (j + 0.5) * cellSize);

                        if (r < 0 || r >= height || c < 0 || c >= width)
                            hist.at<double>(i, j, k) = 0;
                        else
                            hist.at<double>(i, j, k) = filteredOrientChannels[k].at<double>(r, c);
                    }
                }
            }

            Descriptor descriptor;
            for (int i = 0; i < cellNum; i++)
                for (int j = 0; j < cellNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        descriptor.Add(hist.at<double>(i, j, k));

            NormTwoNormalize(descriptor.begin(), descriptor.end());
            return descriptor;
        }

        inline Descriptor Test::GetDescriptor2(const vector<Mat>& enlargeChannels, 
            const vector<Mat>& enlargeFilteredChannels,
            const Point& center, const Mat& weights, int blockSize, int cellNum)
        {
            int cellSize = blockSize / cellNum;
            int imageBottom = enlargeFilteredChannels[0].rows - cellSize,
                imageRight = enlargeFilteredChannels[0].cols - cellSize;
            int expectedTop = center.y - blockSize / 2 + cellSize,
                expectedLeft = center.x - blockSize / 2 + cellSize,
                orientNum = enlargeChannels.size();
            int dims[] = { cellNum, cellNum, orientNum };
            Mat hist(3, dims, CV_64F);

            int kernelSize = weights.rows;
            Mat tmp(kernelSize, kernelSize, CV_64F);

            for (int i = 0; i < cellNum; i++)
            {
                for (int j = 0; j < cellNum; j++)
                {
                    for (int k = 0; k < orientNum; k++)
                    {
                        int r = (int)(expectedTop + (i + 0.5) * cellSize),
                            c = (int)(expectedLeft + (j + 0.5) * cellSize);

                        if (r < cellSize || c < cellSize || r >= imageBottom || c >= imageRight)
                        {
                            hist.at<double>(i, j, k) = 0;
                            continue;
                        }

                        tmp = Scalar::all(0);

                        for (int m = 0; m < kernelSize; m++)
                        {
                            for (int n = 0; n < kernelSize; n++)
                            {
                                int y = r - cellSize + m, x = c - cellSize + n;

                                tmp.at<double>(m, n) = enlargeChannels[k].at<double>(y, x) - 
                                    enlargeFilteredChannels[k].at<double>(r, c);
                                tmp.at<double>(m, n) *= tmp.at<double>(m, n);
                            }
                        }

                        hist.at<double>(i, j, k) = sqrt(tmp.dot(weights));
                    }
                }
            }

            Descriptor descriptor;
            for (int i = 0; i < cellNum; i++)
                for (int j = 0; j < cellNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        descriptor.Add(hist.at<double>(i, j, k));

            NormTwoNormalize(descriptor.begin(), descriptor.end());
            return descriptor;
        }

    }
}
}