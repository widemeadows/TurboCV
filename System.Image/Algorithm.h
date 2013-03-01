#pragma once

#include "../System/Math.h"
#include "Contour.h"
#include "Filter.h"
#include "Geometry.h"
#include "Morphology.h"
#include "Util.h"
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
            static Mat Preprocess(const Mat& sketchImage, bool thinning = false);

            Feature GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false) const;

            Feature GetFeatureWithoutPreprocess(const Mat& sketchImage) const;
            
        protected:
            static Mat GetBoundingBox(const Mat& sketchImage);

            virtual Feature GetFeature(const Mat& sketchImage) const = 0;
        };

        inline Feature Algorithm::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning) const
        {
            return GetFeature(Preprocess(sketchImage, thinning));
        }

        inline Feature Algorithm::GetFeatureWithoutPreprocess(const Mat& sketchImage) const
        {
            return GetFeature(sketchImage);
        }

        inline Mat Algorithm::GetBoundingBox(const Mat& sketchImage)
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

        inline Mat Algorithm::Preprocess(const Mat& sketchImage, bool thinning)
        {
            Mat revImage = reverse(sketchImage);

            Mat cleanedImage;
            clean(revImage, cleanedImage, 3);

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
            resize(squareImage, scaledImage, Size(224, 224));

            Mat paddedImage;
            copyMakeBorder(scaledImage, paddedImage, 16, 16, 16, 16, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
            assert(paddedImage.rows == 256 && paddedImage.cols == 256);

            Mat finalImage;
            if (thinning)
            {
                Mat binaryImage, thinnedImage;

                threshold(paddedImage, binaryImage, 54, 1, CV_THRESH_BINARY);
                thin(binaryImage, thinnedImage);
                threshold(thinnedImage, finalImage, 0.5, 255, CV_THRESH_BINARY);
            }
            else
                finalImage = paddedImage;

            return finalImage;
        }

        ///////////////////////////////////////////////////////////////////////

        class HOG : public Algorithm
        {
        protected:
            virtual Feature GetFeature(const Mat& sketchImage) const;
            
        private:
            static Descriptor GetDescriptor(const vector<Mat>& filteredOrientImages, 
                const Point& centre, int blockSize, int cellSize);
        };

        inline Feature HOG::GetFeature(const Mat& sketchImage) const
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

            vector<Mat> orientChannels = Gradient::GetOrientChannels(sketchImage, orientNum);
            vector<Mat> filteredOrientChannels(orientNum);
            for (int i = 0; i < orientNum; i++)
                filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

            Feature feature;
            int height = sketchImage.rows, width = sketchImage.cols;
            int heightStep = height / sampleNum, widthStep = width / sampleNum;
            for (int i = heightStep / 2; i < height; i += heightStep)
            {
                for (int j = widthStep / 2; j < width; j += widthStep)
                {
                    Descriptor desc = GetDescriptor(filteredOrientChannels, 
                        Point(j, i), blockSize, cellSize);
                    feature.push_back(desc);
                }
            }

            return feature;
        }

        inline Descriptor HOG::GetDescriptor(const vector<Mat>& filteredOrientChannels, 
            const Point& centre, int blockSize, int cellSize)
        {
            int height = filteredOrientChannels[0].rows, 
                width = filteredOrientChannels[0].cols;
            int expectedTop = centre.y - blockSize / 2,
                expectedLeft = centre.x - blockSize / 2,
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
        
        ///////////////////////////////////////////////////////////////////////

        class HOOSC : public Algorithm
        {
        protected:
            virtual Feature GetFeature(const Mat& sketchImage) const;
            
        private:
            static Descriptor GetDescriptor(const Mat& orientImage,
                const Point& pivot, const vector<Point>& points,
                const vector<double>& logDistances, int angleNum, int orientNum);
        };

        inline Feature HOOSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.5, 1 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 9, orientNum = 8;

            vector<Point> points = Contour::GetEdgels(sketchImage);
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);

            tuple<Mat, Mat> gradient = Gradient::GetGradient(sketchImage);
            Mat& powerImage = get<0>(gradient);
            Mat& orientImage = get<1>(gradient);

            Feature feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                Descriptor descriptor = GetDescriptor(orientImage, pivots[i], points,
                    logDistances, angleNum, orientNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline Descriptor HOOSC::GetDescriptor(const Mat& orientImage,
            const Point& pivot, const vector<Point>& points,
            const vector<double>& logDistances, int angleNum, int orientNum)
        {
	        int pointNum = points.size();
            assert(pointNum > 1);

            vector<double> distances = Geometry::EulerDistance(pivot, points);
            vector<double> angles = Geometry::Angle(pivot, points);
	        double mean = Math::Sum(distances) / (pointNum - 1); // Except pivot
	        for (int i = 0; i < pointNum; i++)
		        distances[i] /= mean;

            int distanceNum = logDistances.size() - 1;
	        int dims[] = { distanceNum, angleNum, orientNum };
	        Mat bins(3, dims, CV_64F);
            bins = Scalar::all(0);

	        double angleStep = 2 * CV_PI / angleNum;
	        double orientStep = CV_PI / orientNum;
            double sigma = 10;
	        for (int i = 0; i < pointNum; i++)
	        {
		        if (points[i] == pivot)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = angles[i] / angleStep;
                        if (a >= angleNum)
					        a = angleNum - 1;

                        double orient = orientImage.at<double>(points[i].y, points[i].x);
                        int o = orient / orientStep; 
				        if (o >= orientNum)
					        o = orientNum - 1;

				        double value = Math::Gauss(((o + 0.5) * orientStep - orient) * 180 / CV_PI, sigma);
				        bins.at<double>(j, a, o) += value;

				        break;
			        }
		        }
	        }

            Descriptor descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
			        for (int k = 0; k < orientNum; k++)
				        ring.push_back(bins.at<double>(i, j, k));

		        NormOneNormalize(ring);

                for (auto item : ring)
                    descriptor.push_back(item / distanceNum);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class SC : public Algorithm
        {
        protected:
            virtual Feature GetFeature(const Mat& sketchImage) const;
            
        private:
            static Descriptor GetDescriptor(const Point& pivot, const vector<Point>& pivots,
                const vector<double>& logDistances, int angleNum);
        };

        inline Feature SC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 12;

            vector<Point> points = Contour::GetEdgels(sketchImage);
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);

            Feature feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                Descriptor descriptor = GetDescriptor(pivots[i], pivots, logDistances, angleNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline Descriptor SC::GetDescriptor(const Point& pivot, const vector<Point>& pivots,
                const vector<double>& logDistances, int angleNum)
        {
            int pivotNum = pivots.size();
            assert(pivotNum > 1);

            vector<double> distances = Geometry::EulerDistance(pivot, pivots);
            vector<double> angles = Geometry::Angle(pivot, pivots);
	        double mean = Math::Sum(distances) / (pivotNum - 1); // Except pivot
	        for (int i = 0; i < pivotNum; i++)
		        distances[i] /= mean;

            int distanceNum = logDistances.size() - 1;
	        Mat bins = Mat::zeros(distanceNum, angleNum, CV_64F);

	        double angleStep = 2 * CV_PI / angleNum;
	        for (int i = 0; i < pivotNum; i++)
	        {
		        if (pivots[i] == pivot)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = angles[i] / angleStep;
                        if (a >= angleNum)
					        a = angleNum - 1;

				        bins.at<double>(j, a)++;

				        break;
			        }
		        }
	        }

            Descriptor descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
				    ring.push_back(bins.at<double>(i, j));

		        NormOneNormalize(ring);

                for (auto item : ring)
                    descriptor.push_back(item / distanceNum);
	        }

	        return descriptor;
        }
        
        ///////////////////////////////////////////////////////////////////////

        class SHOG : public Algorithm
        {
        protected:
            virtual Feature GetFeature(const Mat& sketchImage) const;
            
        private:
            static Descriptor GetDescriptor(const vector<Mat>& orientChannels,
                const Point& pivot, const vector<Point>& points, int cellNum);
        };

        inline Feature SHOG::GetFeature(const Mat& sketchImage) const
        {
            int orientNum = 4, cellNum = 4;

            vector<Point> points = Contour::GetEdgels(sketchImage);
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);
            vector<Mat> orientChannels = Gradient::GetOrientChannels(sketchImage, orientNum);

            Feature feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                Descriptor descriptor = GetDescriptor(orientChannels, pivots[i], 
                    points, cellNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline Descriptor SHOG::GetDescriptor(const vector<Mat>& orientChannels,
            const Point& pivot, const vector<Point>& points, int cellNum)
        {
            vector<double> distances = Geometry::EulerDistance(pivot, points);
	        double mean = Math::Sum(distances) / (points.size() - 1); // Except pivot
            int blockSize = 1.5 * mean;

            int height = orientChannels[0].rows, 
                width = orientChannels[0].cols,
                orientNum = orientChannels.size();
            int expectedTop = pivot.y - blockSize / 2,
                expectedLeft = pivot.x - blockSize / 2;
            double cellSize = (double)blockSize / cellNum;
            int dims[] = { cellNum, cellNum, orientNum };
            Mat hist(3, dims, CV_64F);
            hist = Scalar::all(0);

            for (int i = expectedTop; i < expectedTop + blockSize; i++)
            {
                for (int j = expectedLeft; j < expectedLeft + blockSize; j++)
                {
                    if (i < 0 || i >= height || j < 0 || j >= width)
                        continue;

                    for (int k = 0; k < orientNum; k++)
                    {
                        if (abs(orientChannels[k].at<double>(i, j)) < EPS)
                            continue;

                        int r = (i - expectedTop) / cellSize, c = (j - expectedLeft) / cellSize;

                        for (int u = -1; u <= 1; u++)
                        {
                            for (int v = -1; v <= 1; v++)
                            {
                                int newR = r + u, newC = c + v;
                                if (newR < 0 || newR >= cellNum || newC < 0 || newC >= cellNum)
                                    continue;

                                double dRatio = 1 - abs(Geometry::EulerDistance(
                                    Point((newC + 0.5) * cellSize, (newR + 0.5) * cellSize),
                                    Point(j - expectedLeft, i - expectedTop))) / cellSize;
                                if (dRatio < 0)
                                    dRatio = 0;

                                hist.at<double>(newR, newC, k) += orientChannels[k].at<double>(i, j) * dRatio;
                            }
                        }
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

        ///////////////////////////////////////////////////////////////////////

        class RHOOSC : public Algorithm
        {
        protected:
            virtual Feature GetFeature(const Mat& sketchImage) const;
            
        private:
            static Descriptor GetDescriptor(const Mat& orientImage, const Point& centre, 
                const vector<Point>& points, const vector<double>& logDistances, int angleNum, int orientNum);
        };

        inline Feature RHOOSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.5, 1 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 9, orientNum = 8, sampleNum = 28;

            Mat orientImage = get<1>(Gradient::GetGradient(sketchImage));
            vector<Point> points = Contour::GetEdgels(sketchImage); 

            Feature feature;
            int height = sketchImage.rows, width = sketchImage.cols;
            int heightStep = height / sampleNum, widthStep = width / sampleNum;
            for (int i = heightStep / 2; i < height; i += heightStep)
            {
                for (int j = widthStep / 2; j < width; j += widthStep)
                {
                    Descriptor desc = GetDescriptor(orientImage, Point(j, i), points,
                        logDistances, angleNum, orientNum);
                    feature.push_back(desc);
                }
            }

            return feature;
        }

        inline Descriptor RHOOSC::GetDescriptor(const Mat& orientImage, const Point& centre,
            const vector<Point>& points, const vector<double>& logDistances, int angleNum, int orientNum)
        {
            int pointNum = points.size();
            assert(pointNum > 1);

            vector<double> distances = Geometry::EulerDistance(centre, points);
            vector<double> angles = Geometry::Angle(centre, points);
	        double mean = Math::Sum(distances) / (pointNum - 1); // Except pivot
	        for (int i = 0; i < pointNum; i++)
		        distances[i] /= mean;

            int distanceNum = logDistances.size() - 1;
	        int dims[] = { distanceNum, angleNum, orientNum };
	        Mat bins(3, dims, CV_64F);
            bins = Scalar::all(0);

	        double angleStep = 2 * CV_PI / angleNum;
	        double orientStep = CV_PI / orientNum;
            double sigma = 10;
	        for (int i = 0; i < pointNum; i++)
	        {
		        if (points[i] == centre)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = angles[i] / angleStep;
                        if (a >= angleNum)
					        a = angleNum - 1;

                        double orient = orientImage.at<double>(points[i].y, points[i].x);
                        int o = orient / orientStep; 
				        if (o >= orientNum)
					        o = orientNum - 1;

				        double value = Math::Gauss(((o + 0.5) * orientStep - orient) * 180 / CV_PI, sigma);
				        bins.at<double>(j, a, o) += value;

				        break;
			        }
		        }
	        }

            Descriptor descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
			        for (int k = 0; k < orientNum; k++)
				        ring.push_back(bins.at<double>(i, j, k));

		        NormOneNormalize(ring);

                for (auto item : ring)
                    descriptor.push_back(item);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class RSC : public Algorithm
        {
        protected:
            virtual Feature GetFeature(const Mat& sketchImage) const;
            
        private:
            static Descriptor GetDescriptor(const Point& centre, const vector<Point>& points, 
                const vector<double>& logDistances, int angleNum);
        };

        inline Feature RSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 12, sampleNum = 28;

            Mat orientImage = get<1>(Gradient::GetGradient(sketchImage));
            vector<Point> points = Contour::GetEdgels(sketchImage); 

            Feature feature;
            int height = sketchImage.rows, width = sketchImage.cols;
            int heightStep = height / sampleNum, widthStep = width / sampleNum;
            for (int i = heightStep / 2; i < height; i += heightStep)
            {
                for (int j = widthStep / 2; j < width; j += widthStep)
                {
                    Descriptor desc = GetDescriptor(Point(j, i), points, logDistances, angleNum);
                    feature.push_back(desc);
                }
            }

            return feature;
        }

        inline Descriptor RSC::GetDescriptor(const Point& centre, const vector<Point>& points, 
            const vector<double>& logDistances, int angleNum)
        {
            int pointNum = points.size();
            assert(pointNum > 1);

            vector<double> distances = Geometry::EulerDistance(centre, points);
            vector<double> angles = Geometry::Angle(centre, points);
	        double mean = Math::Sum(distances) / (pointNum - 1); // Except pivot
	        for (int i = 0; i < pointNum; i++)
		        distances[i] /= mean;

            int distanceNum = logDistances.size() - 1;
	        Mat bins = Mat::zeros(distanceNum, angleNum, CV_64F);

	        double angleStep = 2 * CV_PI / angleNum;
            double sigma = 10;
	        for (int i = 0; i < pointNum; i++)
	        {
		        if (points[i] == centre)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = angles[i] / angleStep;
                        if (a >= angleNum)
					        a = angleNum - 1;

				        bins.at<double>(j, a)++;

				        break;
			        }
		        }
	        }

            Descriptor descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
				    ring.push_back(bins.at<double>(i, j));

		        NormOneNormalize(ring);

                for (auto item : ring)
                    descriptor.push_back(item / distanceNum);
	        }

	        return descriptor;
        }
    }
}