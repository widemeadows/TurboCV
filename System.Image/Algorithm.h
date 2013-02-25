#pragma once

#include "Util.h"
#include "Filter.h"
#include "Morphology.h"
#include <cv.h>
#include <tuple>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        class Algorithm
        {
        public:
            Feature GetFeature(const Mat& sketchImage, bool thinning = false) const;
            
        protected:
            static Mat GetBoundingBox(const Mat& sketchImage);
            static Mat Preprocess(const Mat& sketchImage, bool thinning = false);

            virtual Feature ComputeFeature(const Mat& sketchImage) const = 0;
        };

        inline Mat Algorithm::GetBoundingBox(const Mat& sketchImage)
        {
            int minX = sketchImage.cols - 1, maxX = 0,
		        minY = sketchImage.rows - 1, maxY = 0;

	        for (int i = 0; i < sketchImage.rows; i++)
		        for (int j = 0; j < sketchImage.cols; j++)
		        {
			        if (!sketchImage.at<uchar>(i, j))
			        {
				        minX = min(minX, j);
				        maxX = max(maxX, j);
				        minY = min(minY, i);
				        maxY = max(maxY, i);
			        }
		        }

	        return Mat(sketchImage, Range(minY, maxY + 1), Range(minX, maxX + 1));
        }

        inline Mat Algorithm::Preprocess(const Mat& sketchImage, bool thinning)
        {
            Mat boundingBox = GetBoundingBox(sketchImage);

            Mat squareImage;
	        int widthPadding = 0, heightPadding = 0;
	        if (boundingBox.rows < boundingBox.cols)
		        heightPadding = (boundingBox.cols - boundingBox.rows) / 2;
	        else
		        widthPadding = (boundingBox.rows - boundingBox.cols) / 2;
	        copyMakeBorder(boundingBox, squareImage, heightPadding, heightPadding, 
                widthPadding, widthPadding, BORDER_CONSTANT, Scalar(255, 255, 255, 255));

            Mat scaledImage;
	        resize(squareImage, scaledImage, Size(224, 224));

	        Mat paddedImage;
	        copyMakeBorder(scaledImage, paddedImage, 16, 16, 16, 16, BORDER_CONSTANT, Scalar(255, 255, 255, 255));
	        assert(paddedImage.rows == 256 && paddedImage.cols == 256);

	        Mat finalImage;
	        if (thinning)
	        {
		        Mat binary, thinned;

                threshold(paddedImage, binary, 200, 1, CV_THRESH_BINARY_INV);
                thin(binary, thinned);
                threshold(thinned, finalImage, 0.5, 255, CV_THRESH_BINARY_INV);
	        }
	        else
		        finalImage = paddedImage;

	        return finalImage;
        }

        inline Feature Algorithm::GetFeature(const Mat& sketchImage, bool thinning) const
        {
            return ComputeFeature(Preprocess(sketchImage, thinning));
        }

        class HOG : public Algorithm
        {
        protected:
            virtual Feature ComputeFeature(const Mat& sketchImage) const;
            
        private:
			vector<Mat> GetOrientChannels(const Mat& sketchImage, int orientNum) const;
            Descriptor ComputeDescriptor(const vector<Mat>& filteredOrientImages, int centreY, int centreX, 
                int blockSize, int cellSize) const;
        };

		inline vector<Mat> HOG::GetOrientChannels(const Mat& sketchImage, int orientNum) const
		{
			tuple<Mat, Mat> gradient = ComputeGradient(sketchImage);
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

				        orientChannels[newO].at<double>(i, j) += powerImage.at<double>(i, j) * oRatio;
			        }
		        }
			}

			return orientChannels;
		}

        inline Descriptor HOG::ComputeDescriptor(const vector<Mat>& filteredOrientChannels, 
            int centreY, int centreX, int blockSize, int cellSize) const
        {
	        int height = filteredOrientChannels[0].rows, 
		        width = filteredOrientChannels[0].cols;
	        int expectedTop = centreY - blockSize / 2,
		        expectedLeft = centreX - blockSize / 2,
		        cellHalfSize = cellSize / 2,
                cellNum = blockSize / cellSize,
                orientNum = filteredOrientChannels.size();
	        int dims[] = { cellNum, cellNum, orientNum };
	        Mat hist(3, dims, CV_64F);

	        for (int i = 0; i < cellNum; i++)
			{
		        for (int j = 0; j < cellNum; j++)
				{
			        for (int k = 0; k < orientNum; k++)
			        {
				        int r = expectedTop + i * cellSize + cellHalfSize,
					        c = expectedLeft + j * cellSize + cellHalfSize;

				        if (r < 0 || r >= height || c < 0 || c >= width)
					        hist.at<double>(i, j, k) = 0;
				        else
					        hist.at<double>(i, j, k) = filteredOrientChannels[k].at<double>(r, c);
			        }
				}
			}

	        Descriptor desc;
	        for (int i = 0; i < cellNum; i++)
		        for (int j = 0; j < cellNum; j++)
			        for (int k = 0; k < orientNum; k++)
				        desc.push_back(hist.at<double>(i, j, k));

	        NormTwoNormalize(desc);
	        return desc;
        }

		inline Feature HOG::ComputeFeature(const Mat& sketchImage) const
        {
            int orientNum = 4, cellNum = 4, blockSize = 92, sampleNum = 28;
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

			vector<Mat> orientChannels = GetOrientChannels(sketchImage, orientNum);
	        vector<Mat> filteredOrientChannels(orientNum);
	        for (int i = 0; i < orientNum; i++)
		        filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

	        Feature feature;
			int height = sketchImage.rows, width = sketchImage.cols;
	        int heightStep = (height / sampleNum + 0.5), 
				widthStep = (width / sampleNum + 0.5);
	        for (int i = heightStep / 2; i < height; i += heightStep)
			{
		        for (int j = widthStep / 2; j < width; j += widthStep)
		        {
			        Descriptor desc = ComputeDescriptor(filteredOrientChannels, i, j, blockSize, cellSize);
                    feature.push_back(desc);
		        }
			}

	        return feature;
        }
    }
}