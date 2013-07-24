#include "../System/System.h"
#include "Core.h"
#include <cv.h>
#include <highgui.h>
#include <cassert>
#include <unordered_set>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
	namespace Image
	{
        //////////////////////////////////////////////////////////////////////////
        // APIs for Basic Operations
        //////////////////////////////////////////////////////////////////////////

		Mat reverse(Mat image) 
		{
			assert(image.type() == CV_8U);
			Mat result(image.size(), CV_8U);

			for (int i = 0; i < image.rows; i++)
				for (int j = 0; j < image.cols; j++)
					result.at<uchar>(i, j) = 255 - image.at<uchar>(i, j);

			return result;
		}

        Mat FFTShift(const Mat& image)
        {
            int width = image.cols, height = image.rows;
            Mat result(image.rows, image.cols, image.type());

            for (int i = 0; i < height / 2; i++)
            {
                for (int j = 0; j < width / 2; j++)
                    result.at<double>(i, j) = image.at<double>(i + (height + 1) / 2, j + (width + 1) / 2);

                for (int j = 0; j < (width + 1) / 2; j++)
                    result.at<double>(i, j + width / 2) = image.at<double>(i + (height + 1) / 2, j);
            }

            for (int i = 0; i < (height + 1) / 2; i++)
            {
                for (int j = 0; j < width / 2; j++)
                    result.at<double>(i + height / 2, j) = image.at<double>(i, j + (width + 1) / 2);

                for (int j = 0; j < (width + 1) / 2; j++)
                    result.at<double>(i + height / 2, j + width / 2) = image.at<double>(i, j);
            }

            return result;
        }

        void ConvolveDFT(InputArray src, OutputArray dst, int ddepth, InputArray kernel)
        {
            Mat A = src.getMat(), B = kernel.getMat();
            if (ddepth != -1)
            {
                A.convertTo(A, ddepth);
                B.convertTo(B, ddepth);
            }
            CV_Assert(B.rows % 2 != 0 && B.cols % 2 != 0);

            int actualType = ddepth == -1 ? A.type() : ddepth;

            // reallocate the output array if needed
            dst.create(A.rows, A.cols, actualType);
            Mat C = dst.getMat();

            Size imgSize = Size(A.cols + B.cols - 1, A.rows + B.rows - 1);

            // calculate the size of DFT transform
            Size dftSize;
            dftSize.width = getOptimalDFTSize(imgSize.width);
            dftSize.height = getOptimalDFTSize(imgSize.height);

            // allocate temporary buffers and initialize them
            Mat tempA = Mat::zeros(dftSize, actualType);
            Mat roiA(tempA, Rect(B.cols / 2, B.rows / 2, A.cols, A.rows));
            A.copyTo(roiA);
            copyMakeBorder(roiA, tempA(Rect(0, 0, imgSize.width, imgSize.height)), 
                B.rows / 2, B.rows / 2, B.cols / 2, B.cols / 2, 
                BORDER_DEFAULT | BORDER_ISOLATED);

            Mat tempB = Mat::zeros(dftSize, actualType);
            Mat roiB(tempB, Rect(0, 0, B.cols, B.rows));
            B.copyTo(roiB);

            // now transform the padded A & B in-place;
            // use "nonzeroRows" hint for faster processing
            dft(tempA, tempA, 0, imgSize.height);
            dft(tempB, tempB, 0, B.rows);

            // multiply the spectrums;
            // the function handles packed spectrum representations well
            mulSpectrums(tempA, tempB, tempA, 0, true);

            // transform the product back from the frequency domain.
            // Even though all the result rows will be non-zero,
            // you need only the first C.rows of them, and thus you
            // pass nonzeroRows == C.rows
            dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

            // now copy the result back to C.
            tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
        }

        Mat imshow(const Mat& image, bool scale)
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
                        tmp.at<double>(i, j) = (tmp.at<double>(i, j) - minimum) / (maximum - minimum);
            }

            imshow("OpenCV", tmp);
            return tmp;
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for Gradient
        //////////////////////////////////////////////////////////////////////////
        
        Group<Mat, Mat> GetGradientKernel(double sigma, double epsilon)
        {
            int halfSize = (int)ceil(sigma * sqrt(-2 * log(sqrt(2 * CV_PI) * sigma * epsilon)));
            int size = halfSize * 2 + 1;
            double sum = 0, root;
            Mat dx(size, size, CV_64F), dy(size, size, CV_64F);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dx.at<double>(i, j) = Math::Gauss(i - halfSize, sigma) * 
                        Math::GaussDeriv(j - halfSize, sigma);
                    dy.at<double>(j, i) = dx.at<double>(i, j);
                    sum += dx.at<double>(i, j) * dx.at<double>(i, j);
                }
            }

            root = sqrt(sum);
            if (root > 0)
            {
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        dx.at<double>(i, j) /= root;
                        dy.at<double>(i, j) /= root;
                    }
                }
            }

            return CreateGroup(dx, dy);
        }

        Group<Mat, Mat> GetGradient(const Mat& image, double sigma)
        {
            Group<Mat, Mat> kernel = GetGradientKernel(sigma, 1e-2);
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
                    powerImage.at<double>(i, j) = sqrt(
                    dyImage.at<double>(i, j) * dyImage.at<double>(i, j) +
                    dxImage.at<double>(i, j) * dxImage.at<double>(i, j));

            return CreateGroup(powerImage, orientImage);
        }

        ArrayList<Mat> GetOrientChannels(const Mat& image, int orientNum)
        {
            Group<Mat, Mat> gradient = GetGradient(image);
            Mat& powerImage = gradient.Item1();
            Mat& orientImage = gradient.Item2();
            int height = image.rows, width = image.cols;
            double orientBinSize = CV_PI / orientNum;

            ArrayList<Mat> orientChannels;
            for (int i = 0; i < orientNum; i++)
                orientChannels.Add(Mat::zeros(height, width, CV_64F));

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


        //////////////////////////////////////////////////////////////////////////
        // APIs for LoG and DoG
        //////////////////////////////////////////////////////////////////////////

        // http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
        Mat GetLoGKernel(int ksize, double sigma, int ktype)
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

        ArrayList<Mat> GetLoGPyramid(const Mat& image, const ArrayList<double>& sigmas)
        {
            size_t sigmaNum = sigmas.Count();
            ArrayList<Mat> LoGPyramid(sigmaNum);

            for (int i = 0; i < sigmaNum; i++)
            {
                CV_Assert(sigmas[i] >= 0);

                int ksize = (int)(sigmas[i] * 6 + 1);
                if (ksize % 2 == 0)
                    ksize++;

                Mat kernel = GetLoGKernel(ksize, sigmas[i], CV_64F);
                filter2D(image, LoGPyramid[i], CV_64F, kernel);
                LoGPyramid[i] = abs(LoGPyramid[i]) * pow(sigmas[i], 4); // pow(sigmas[i], 4) normalizes the integral
            }

            return LoGPyramid;
        }

        ArrayList<Mat> GetDoGPyramid(const Mat& image, const ArrayList<double>& sigmas)
        {
            size_t sigmaNum = sigmas.Count();
            ArrayList<Mat> GaussianPyramid(sigmaNum + 1);

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

            ArrayList<Mat> DoGPyramid;
            for (int i = 1; i <= sigmaNum; i++)
                DoGPyramid[i - 1] = GaussianPyramid[i] - GaussianPyramid[i - 1];

            return DoGPyramid;
        }


        //////////////////////////////////////////////////////////////////////////
        // Others
        //////////////////////////////////////////////////////////////////////////

        ArrayList<Point> SampleOnGrid(size_t height, size_t width, size_t numPerDirection)
        {
            int heightStep = height / numPerDirection, 
                widthStep = width / numPerDirection;
            ArrayList<Point> points;

            for (int i = heightStep / 2; i < height; i += heightStep)
                for (int j = widthStep / 2; j < width; j += widthStep)
                    points.Add(Point(j, i));

            return points;
        }

        ArrayList<Point> SampleFromPoints(const ArrayList<Point>& points, size_t samplingNum)
        {
            size_t pointNum = points.Count();
            assert(pointNum >= samplingNum);

            ArrayList<Group<double, Group<size_t, size_t>>> distances(pointNum * (pointNum - 1) / 2);
            unordered_set<Point, PointHash> pivots;

            int counter = 0;
            for (size_t i = 0; i < pointNum; i++)
            {
                for (size_t j = i + 1; j < pointNum; j++)
                {
                    double distance = EulerDistance(points[i], points[j]);
                    distances[counter++] = CreateGroup(distance, CreateGroup(i, j));
                }
                pivots.insert(points[i]);
            }
            sort(distances.begin(), distances.end());

            int ptr = 0;
            while (pivots.size() > samplingNum)
            {
                Group<size_t, size_t> idxPair = distances[ptr++].Item2();
                Group<Point, Point> pointPair = CreateGroup(points[idxPair.Item1()], points[idxPair.Item2()]);

                if (pivots.find(pointPair.Item1()) != pivots.end() &&
                    pivots.find(pointPair.Item2()) != pivots.end())
                    pivots.erase(pointPair.Item2());
            }

            ArrayList<Point> results;
            for (auto pivot : pivots)
                results.Add(pivot);

            return results;
        }
	}
}
}