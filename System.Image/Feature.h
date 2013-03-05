#pragma once

#include "../System/String.h"
#include "../System/Math.h"
#include "Contour.h"
#include "Filter.h"
#include "Geometry.h"
#include "Info.h"
#include "Morphology.h"
#include <cv.h>
#include <tuple>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        class Feature
        {
        public:
            static Mat Preprocess(const Mat& sketchImage, bool thinning = false);

            FeatureInfo<double> GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false) const;

            FeatureInfo<double> GetFeatureWithoutPreprocess(const Mat& sketchImage) const;

            virtual String GetName() const = 0;
            
        protected:
            static Mat GetBoundingBox(const Mat& sketchImage);

            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const = 0;
        };

        inline FeatureInfo<double> Feature::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning) const
        {
            return GetFeature(Preprocess(sketchImage, thinning));
        }

        inline FeatureInfo<double> Feature::GetFeatureWithoutPreprocess(const Mat& sketchImage) const
        {
            return GetFeature(sketchImage);
        }

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

        class HOG : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "hog"; };

        private:
            static DescriptorInfo<double> GetDescriptor(const vector<Mat>& filteredOrientImages, 
                const Point& centre, int blockSize, int cellNum);
        };

        inline FeatureInfo<double> HOG::GetFeature(const Mat& sketchImage) const
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

            FeatureInfo<double> feature;
            int height = sketchImage.rows, width = sketchImage.cols;
            int heightStep = height / sampleNum, widthStep = width / sampleNum;
            for (int i = heightStep / 2; i < height; i += heightStep)
            {
                for (int j = widthStep / 2; j < width; j += widthStep)
                {
                    DescriptorInfo<double> desc = GetDescriptor(filteredOrientChannels, 
                        Point(j, i), blockSize, cellNum);
                    feature.push_back(desc);
                }
            }

            return feature;
        }

        inline DescriptorInfo<double> HOG::GetDescriptor(const vector<Mat>& filteredOrientChannels, 
            const Point& centre, int blockSize, int cellNum)
        {
            int height = filteredOrientChannels[0].rows, 
                width = filteredOrientChannels[0].cols;
            int expectedTop = centre.y - blockSize / 2,
                expectedLeft = centre.x - blockSize / 2,
                cellSize = blockSize / cellNum,
                cellHalfSize = cellSize / 2,
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

            DescriptorInfo<double> desc;
            for (int i = 0; i < cellNum; i++)
                for (int j = 0; j < cellNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        desc.push_back(hist.at<double>(i, j, k));

            NormTwoNormalize(desc.begin(), desc.end());
            return desc;
        }
        
        ///////////////////////////////////////////////////////////////////////

        class HOOSC : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "hoosc"; };

        private:
            static DescriptorInfo<double> GetDescriptor(const Mat& orientImage,
                const Point& pivot, const vector<Point>& points,
                const vector<double>& logDistances, int angleNum, int orientNum);
        };

        inline FeatureInfo<double> HOOSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.5, 1 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 9, orientNum = 8;

            vector<Point> points = Contour::GetEdgels(sketchImage);
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);

            tuple<Mat, Mat> gradient = Gradient::GetGradient(sketchImage);
            Mat& powerImage = get<0>(gradient);
            Mat& orientImage = get<1>(gradient);

            FeatureInfo<double> feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                DescriptorInfo<double> descriptor = GetDescriptor(orientImage, pivots[i], points,
                    logDistances, angleNum, orientNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline DescriptorInfo<double> HOOSC::GetDescriptor(const Mat& orientImage,
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

            DescriptorInfo<double> descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
			        for (int k = 0; k < orientNum; k++)
				        ring.push_back(bins.at<double>(i, j, k));

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item / distanceNum);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class SC : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "sc"; };

        private:
            static DescriptorInfo<double> GetDescriptor(const Point& pivot, const vector<Point>& pivots,
                const vector<double>& logDistances, int angleNum);
        };

        inline FeatureInfo<double> SC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 12;

            vector<Point> points = Contour::GetEdgels(sketchImage);
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);

            FeatureInfo<double> feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                DescriptorInfo<double> descriptor = GetDescriptor(pivots[i], pivots, 
                    logDistances, angleNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline DescriptorInfo<double> SC::GetDescriptor(const Point& pivot, const vector<Point>& pivots,
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

            DescriptorInfo<double> descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
				    ring.push_back(bins.at<double>(i, j));

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item / distanceNum);
	        }

	        return descriptor;
        }
        
        ///////////////////////////////////////////////////////////////////////

        class SHOG : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "shog"; };

        private:
            static DescriptorInfo<double> GetDescriptor(const vector<Mat>& orientChannels,
                const Point& pivot, const vector<Point>& points, int cellNum);
        };

        inline FeatureInfo<double> SHOG::GetFeature(const Mat& sketchImage) const
        {
            int orientNum = 4, cellNum = 4;

            vector<Point> points = Contour::GetEdgels(sketchImage);
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);
            vector<Mat> orientChannels = Gradient::GetOrientChannels(sketchImage, orientNum);

            FeatureInfo<double> feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                DescriptorInfo<double> descriptor = GetDescriptor(orientChannels, pivots[i], 
                    points, cellNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline DescriptorInfo<double> SHOG::GetDescriptor(const vector<Mat>& orientChannels,
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

            DescriptorInfo<double> desc;
            for (int i = 0; i < cellNum; i++)
                for (int j = 0; j < cellNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        desc.push_back(hist.at<double>(i, j, k));

            NormTwoNormalize(desc.begin(), desc.end());
            return desc;
        }

        ///////////////////////////////////////////////////////////////////////

        class RHOOSC : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "rhoosc"; };

        private:
            static DescriptorInfo<double> GetDescriptor(const Mat& orientImage, 
                const Point& centre, const vector<Point>& points, 
                const vector<double>& logDistances, int angleNum, int orientNum);
        };

        inline FeatureInfo<double> RHOOSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.5, 1 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 9, orientNum = 8, sampleNum = 28;

            Mat orientImage = get<1>(Gradient::GetGradient(sketchImage));
            vector<Point> points = Contour::GetEdgels(sketchImage); 

            FeatureInfo<double> feature;
            int height = sketchImage.rows, width = sketchImage.cols;
            int heightStep = height / sampleNum, widthStep = width / sampleNum;
            for (int i = heightStep / 2; i < height; i += heightStep)
            {
                for (int j = widthStep / 2; j < width; j += widthStep)
                {
                    DescriptorInfo<double> desc = GetDescriptor(orientImage, Point(j, i), points,
                        logDistances, angleNum, orientNum);
                    feature.push_back(desc);
                }
            }

            return feature;
        }

        inline DescriptorInfo<double> RHOOSC::GetDescriptor(const Mat& orientImage, 
            const Point& centre, const vector<Point>& points, 
            const vector<double>& logDistances, int angleNum, int orientNum)
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

            DescriptorInfo<double> descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
			        for (int k = 0; k < orientNum; k++)
				        ring.push_back(bins.at<double>(i, j, k));

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class RSC : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "rsc"; };
            
        private:
            static DescriptorInfo<double> GetDescriptor(const Point& centre, 
                const vector<Point>& pivots, const vector<double>& logDistances, int angleNum);
        };

        inline FeatureInfo<double> RSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 12, sampleNum = 28;

            Mat orientImage = get<1>(Gradient::GetGradient(sketchImage));
            vector<Point> points = Contour::GetEdgels(sketchImage); 
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);

            FeatureInfo<double> feature;
            int height = sketchImage.rows, width = sketchImage.cols;
            int heightStep = height / sampleNum, widthStep = width / sampleNum;
            for (int i = heightStep / 2; i < height; i += heightStep)
            {
                for (int j = widthStep / 2; j < width; j += widthStep)
                {
                    DescriptorInfo<double> desc = GetDescriptor(Point(j, i), pivots, logDistances, angleNum);
                    feature.push_back(desc);
                }
            }

            return feature;
        }

        inline DescriptorInfo<double> RSC::GetDescriptor(const Point& centre, 
            const vector<Point>& pivots, const vector<double>& logDistances, int angleNum)
        {
            int pivotNum = pivots.size();
            assert(pivotNum > 1);

            vector<double> distances = Geometry::EulerDistance(centre, pivots);
            vector<double> angles = Geometry::Angle(centre, pivots);
	        double mean = Math::Sum(distances) / (pivotNum - 1); // Except pivot
	        for (int i = 0; i < pivotNum; i++)
		        distances[i] /= mean;

            int distanceNum = logDistances.size() - 1;
	        Mat bins = Mat::zeros(distanceNum, angleNum, CV_64F);

	        double angleStep = 2 * CV_PI / angleNum;
            double sigma = 10;
	        for (int i = 0; i < pivotNum; i++)
	        {
		        if (pivots[i] == centre)
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

            DescriptorInfo<double> descriptor;
	        for (int i = 0; i < distanceNum; i++)
	        {
                vector<double> ring;
		        for (int j = 0; j < angleNum; j++)
				    ring.push_back(bins.at<double>(i, j));

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class ASHOG : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "ashog"; };
            
        private:
            static DescriptorInfo<double> GetDescriptor(const vector<Mat>& orientChannels, 
                const Point& pivot, int blockSize, int cellNum);
        };

        inline FeatureInfo<double> ASHOG::GetFeature(const Mat& sketchImage) const
        {
            int orientNum = 4, cellNum = 4, scaleNum = 15;
            double sigmaInit = 0.7, sigmaStep = 1.2;
            
            vector<double> sigmas;
            sigmas.push_back(sigmaInit);
            for (int i = 1; i < scaleNum; i++)
                sigmas.push_back(sigmas[i - 1] * sigmaStep);

            vector<Point> points = Contour::GetEdgels(sketchImage);
            vector<Point> pivots = Contour::GetPivots(points, points.size() * 0.33);
            vector<Mat> orientChannels = Gradient::GetOrientChannels(sketchImage, orientNum);
            vector<Mat> pyramid = GetLoGPyramid(sketchImage, sigmas);
            
            FeatureInfo<double> feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                for (int j = 0; j < scaleNum; j++)
                {
                    double prev = j > 0 ? pyramid[j - 1].at<double>(pivots[i].y, pivots[i].x) : 0;
                    double curr = pyramid[j].at<double>(pivots[i].y, pivots[i].x);
                    double next = j < scaleNum - 1 ? pyramid[j + 1].at<double>(pivots[i].y, pivots[i].x) : 0;

                    if (curr > next && curr > prev)
                    {
                        DescriptorInfo<double> descriptor = GetDescriptor(orientChannels, 
                            pivots[i], sigmas[j] * 6, cellNum);
                        feature.push_back(descriptor);
                    }
                }
            }

            return feature;
        }

        inline DescriptorInfo<double> ASHOG::GetDescriptor(const vector<Mat>& orientChannels, 
            const Point& pivot, int blockSize, int cellNum)
        {
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

            DescriptorInfo<double> desc;
            for (int i = 0; i < cellNum; i++)
                for (int j = 0; j < cellNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        desc.push_back(hist.at<double>(i, j, k));

            NormTwoNormalize(desc.begin(), desc.end());
            return desc;
        }

        ///////////////////////////////////////////////////////////////////////

        class Gabor : public Feature
        {
        protected:
            virtual FeatureInfo<double> GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "gabor"; };
            
        private:
            static DescriptorInfo<double> GetDescriptor(const vector<Mat>& gaborResponses, 
                const Point& centre, int blockSize, int cellNum);
        };

        inline FeatureInfo<double> Gabor::GetFeature(const Mat& sketchImage) const
        {
            int sampleNum = 28, blockSize = 92, cellNum = 4;
            int tmp[] = { 8, 8, 8, 8 };
            vector<int> orientNumPerScale(tmp, tmp + sizeof(tmp) / sizeof(int));

            vector<Mat> gaborResponses;
            for (int i = 0; i < orientNumPerScale.size(); i++)
            {
                double sigma = pow(1.8, i + 1), lambda = sigma * 1.7;

                int ksize = sigma * 6 + 1;
                if (ksize % 2 == 0)
                    ksize++;

                for (int j = 0; j < orientNumPerScale[i]; j++)
                {
                    Mat kernel = getGaborKernel(Size(ksize, ksize), sigma, 
                        CV_PI / orientNumPerScale[i] * j, lambda, 1, 0);

                    Mat gaborResponse;
                    filter2D(sketchImage, gaborResponse, CV_64F, kernel);

			        gaborResponses.push_back(abs(gaborResponse));
                }
            }

            FeatureInfo<double> feature;
            int height = sketchImage.rows, width = sketchImage.cols;
            int heightStep = height / sampleNum, widthStep = width / sampleNum;
            for (int i = heightStep / 2; i < height; i += heightStep)
            {
		        for (int j = widthStep / 2; j < width; j += widthStep)
		        {
			        DescriptorInfo<double> descriptor = GetDescriptor(gaborResponses,
                        Point(j, i), blockSize, cellNum);
                    feature.push_back(descriptor);
		        }
            }

            return feature;
        }

        inline DescriptorInfo<double> Gabor::GetDescriptor(const vector<Mat>& gaborResponses, 
                const Point& centre, int blockSize, int cellNum)
        {
            assert(gaborResponses.size() > 0);

            int height = gaborResponses[0].rows,
                width = gaborResponses[0].cols,
                expectedTop = centre.y - blockSize / 2,
                expectedLeft = centre.x - blockSize / 2,
                cellSize = blockSize / cellNum;

            DescriptorInfo<double> desc;
            for (int i = 0; i < gaborResponses.size(); i++)
            {
                vector<double> cells(cellNum * cellNum);

                for (int j = expectedTop; j < expectedTop + blockSize; j++)
                {
                    for (int k = expectedLeft; k < expectedLeft + blockSize; k++)
                    {
                        if (j < 0 || j >= height || k < 0 || k >= width)
                            continue;

                        int cellRow = (j - expectedTop) / cellSize,
				            cellCol = (k - expectedLeft) / cellSize;
			            cells[cellRow * cellNum + cellCol] += gaborResponses[i].at<double>(j, k);
                    }
                }

                for (int j = 0; j < cells.size(); j++)
                    desc.push_back(cells[i]);
            }

            NormTwoNormalize(desc.begin(), desc.end());
            return desc;
        }
    }
}