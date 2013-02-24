#pragma once

#include "System.Image.h"
#include <cv.h>
#include <tuple>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        typedef vector<double> Descriptor;

        class Feature
        {
        public:
            void GetFeature(const Mat& sketchImage, bool thinning = false);
            
        protected:
            static Mat GetBoundingBox(const Mat& sketchImage);
            static Mat Preprocess(const Mat& sketchImage, bool thinning = false);

            virtual vector<Descriptor> ComputeFeature(const Mat& sketchImage) = 0;
        };

        inline Mat Feature::GetBoundingBox(const Mat& sketchImage)
        {
            int minX = sketchImage.cols - 1, maxX = 0,
		        minY = sketchImage.rows - 1, maxY = 0;

	        for (int i = 0; i < sketchImage.rows; i++)
		        for (int j = 0; j < sketchImage.cols; j++)
		        {
			        if (sketchImage.at<uchar>(i, j))
			        {
				        minX = min(minX, j);
				        maxX = max(maxX, j);
				        minY = min(minY, i);
				        maxY = max(maxY, i);
			        }
		        }

	        return Mat(sketchImage, Range(minY, maxY + 1), Range(minX, maxX + 1));
        }

        inline Mat Feature::Preprocess(const Mat& sketchImage, bool thinning)
        {
            Mat revImage = reverse(sketchImage);

            Mat boundingBox = GetBoundingBox(revImage);

            Mat squareImage;
	        int widthPadding = 0, heightPadding = 0;
	        if (boundingBox.rows < boundingBox.cols)
		        heightPadding = (boundingBox.cols - boundingBox.rows) / 2;
	        else
		        widthPadding = (boundingBox.rows - boundingBox.cols) / 2;
	        copyMakeBorder(boundingBox, squareImage, heightPadding, heightPadding, 
                widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

            Mat scaledImage;
	        resize(squareImage, scaledImage, Size(112, 112));

	        Mat paddedImage;
	        copyMakeBorder(scaledImage, paddedImage, 8, 8, 8, 8, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
	        assert(paddedImage.rows == 128 && paddedImage.cols == 128);

	        Mat finalImage;
	        if (thinning)
	        {
		            Mat thre, thinned;
                    
                    threshold(paddedImage, thre, 54, 1, CV_THRESH_BINARY);
                    thin(thre, thinned);
                    threshold(thinned, finalImage, 0.5, 255, CV_THRESH_BINARY);
	        }
	        else
		        finalImage = paddedImage;

	        return finalImage;
        }

        inline void Feature::GetFeature(const Mat& sketchImage, bool thinning)
        {
            Mat image = Preprocess(sketchImage, thinning);

            ComputeFeature(image);
        }


        class HOG : public Feature
        {
        protected:
            virtual vector<Descriptor> ComputeFeature(const Mat& sketchImage);
            
        private:
            Descriptor ComputeDescriptor(const vector<Mat>& filteredOrientImages, int centreY, int centreX, 
                int blockSize, int cellSize);
        };

        inline vector<Descriptor> HOG::ComputeFeature(const Mat& sketchImage)
        {
            tuple<Mat, Mat> gradient = ComputeGradient(sketchImage);
            Mat& powerImage = get<0>(gradient);
            Mat& orientImage = get<1>(gradient);

            int orientNum = 4;

            int height = powerImage.rows, width = powerImage.cols;
	        Mat* orientImages = new Mat[orientNum];
	        double orientBinSize = CV_PI / orientNum;

	        for (int i = 0; i < orientNum; i++)
		        orientImages[i] = Mat::zeros(height, width, CV_64F);

	        for (int i = 0; i < height; i++)
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
				        double oRatio = 1 - abs((newO + 0.5) * orientBinSize - orientImage.at<double>(i, j)) / orientBinSize;
				        if (oRatio < 0)
					        oRatio = 0;
			
				        if (newO == -1)
					        newO = orientNum - 1;
				        if (newO == orientNum)
					        newO = 0;

				        orientImages[newO].at<double>(i, j) += powerImage.at<double>(i, j) * oRatio;
			        }
		        }

                int blockSize = 92;
                int spacialNum = 4;

	            int cellSize = blockSize / spacialNum, kernelSize = cellSize * 2 + 1;
	            Mat tentKernel(kernelSize, kernelSize, CV_64F);
	            for (int i = 0; i < kernelSize; i++)
		            for (int j = 0; j < kernelSize; j++)
		            {
			            double ratio = 1 - sqrt((i - cellSize) * (i - cellSize) + (j - cellSize) * (j - cellSize)) / cellSize;
			            if (ratio < 0)
				            ratio = 0;
			            tentKernel.at<double>(i, j) = ratio;
		            }

	        vector<Mat> filteredOrientBinImages(orientNum);
	        for (int i = 0; i < orientNum; i++)
		        filter2D(orientImages[i], filteredOrientBinImages[i], -1, tentKernel);

            int sampleNum = 28;
	        vector<Descriptor> feature;
	        int heightStep = height / sampleNum, widthStep = width / sampleNum;
	        for (int i = heightStep / 2; i < height; i += heightStep)
		        for (int j = widthStep / 2; j < width; j += widthStep)
		        {
			        Descriptor desc = ComputeDescriptor(filteredOrientBinImages, i, j, blockSize, cellSize);
                    feature.push_back(desc);
		        }

	        delete[] orientImages;

	        return feature;
        }

        inline Descriptor HOG::ComputeDescriptor(const vector<Mat>& filteredOrientImages, 
            int centreY, int centreX, int blockSize, int cellSize)
        {
	        int height = filteredOrientImages[0].rows, 
		        width = filteredOrientImages[0].cols;
	        int expectedTop = centreY - blockSize / 2,
		        expectedLeft = centreX - blockSize / 2,
		        cellHalfSize = cellSize / 2,
                spacialNum = blockSize / cellSize,
                orientNum = filteredOrientImages.size();
	        int dims[] = { spacialNum, spacialNum, orientNum };
	        Mat hist(3, dims, CV_64F);

	        for (int i = 0; i < spacialNum; i++)
		        for (int j = 0; j < spacialNum; j++)
			        for (int k = 0; k < orientNum; k++)
			        {
				        int r = expectedTop + i * blockSize + cellHalfSize,
					        c = expectedLeft + j * blockSize + cellHalfSize;

				        if (r < 0 || r >= height || c < 0 || c >= width)
					        hist.at<double>(i, j, k) = 0;
				        else
					        hist.at<double>(i, j, k) = filteredOrientImages[k].at<double>(r, c);
			        }

	        Descriptor desc;
	        for (int i = 0; i < spacialNum; i++)
		        for (int j = 0; j < spacialNum; j++)
			        for (int k = 0; k < orientNum; k++)
				        desc.push_back(hist.at<double>(i, j, k));

	        NormTwoNormalize(desc);

	        return desc;
        }
    }
}