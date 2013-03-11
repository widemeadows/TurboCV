#pragma once

#include "../System/String.h"
#include "../System/Math.h"
#include "BinaryImage.h"
#include "Filter.h"
#include "Geometry.h"
#include "Sample.h"
#include "Typedef.h"
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

            virtual String GetName() const = 0;
            
        protected:
            static Mat GetBoundingBox(const Mat& sketchImage);
        };

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

        ///////////////////////////////////////////////////////////////////////

        class LocalFeature : public Feature
        {
        public:
            LocalFeatureVec GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false) const;
            LocalFeatureVec GetFeatureWithoutPreprocess(const Mat& sketchImage) const;

        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const = 0;
        };

        inline LocalFeatureVec LocalFeature::GetFeatureWithPreprocess(const Mat& sketchImage, 
            bool thinning) const
        {
            return GetFeature(Preprocess(sketchImage, thinning));
        }

        inline LocalFeatureVec LocalFeature::GetFeatureWithoutPreprocess(const Mat& sketchImage) const
        {
            return GetFeature(sketchImage);
        }

        ///////////////////////////////////////////////////////////////////////

        class GlobalFeature : public Feature
        {
        public:
            GlobalFeatureVec GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false) const;
            GlobalFeatureVec GetFeatureWithoutPreprocess(const Mat& sketchImage) const;

        protected:
            virtual GlobalFeatureVec GetFeature(const Mat& sketchImage) const = 0;
        };

        inline GlobalFeatureVec GlobalFeature::GetFeatureWithPreprocess(const Mat& sketchImage, 
            bool thinning) const
        {
            return GetFeature(Preprocess(sketchImage, thinning));
        }

        inline GlobalFeatureVec GlobalFeature::GetFeatureWithoutPreprocess(const Mat& sketchImage) const
        {
            return GetFeature(sketchImage);
        }

        ///////////////////////////////////////////////////////////////////////

        /*class Test : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "test"; };

        private:
            static Descriptor GetDescriptor(const Size& size, const Point& center, 
                const vector<Point>& points, int blockSize, int cellNum);
        };

        inline LocalFeatureVec Test::GetFeature(const Mat& sketchImage) const
        {
            int sampleNum = 28, blockSize = 112, cellNum = 8;

            vector<Point> points = GetEdgels(sketchImage);

            LocalFeatureVec feature;
            vector<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor descriptor = GetDescriptor(sketchImage.size(), center, points, blockSize, cellNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline Descriptor Test::GetDescriptor(const Size& size, const Point& center, 
            const vector<Point>& points, int blockSize, int cellNum)
        {
            int height = size.height, width = size.width;
            double cellSize = (double)blockSize / cellNum;
            int expectedTop = center.y - blockSize / 2,
                expectedLeft = center.x - blockSize / 2;
            Mat hist = Mat::zeros(cellNum, cellNum, CV_64F);

            for (auto point : points)
            {
                int r = (int)((point.y - expectedTop) / cellSize);
                int c = (int)((point.x - expectedLeft) / cellSize);

                if (r < 0 || r >= cellNum || c < 0 || c >= cellNum)
                    continue;
                else
                    hist.at<double>(r, c)++;
            }

            Descriptor descriptor;
            for (int i = 0; i < cellNum; i++)
                for (int j = 0; j < cellNum; j++)
                    descriptor.push_back(hist.at<double>(i, j));

            NormOneNormalize(descriptor.begin(), descriptor.end());
            return descriptor;
        }*/

        ///////////////////////////////////////////////////////////////////////

        class HOG : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "hog"; };

        private:
            static Descriptor GetDescriptor(const vector<Mat>& filteredOrientImages, 
                const Point& center, int blockSize, int cellNum);
        };

        inline LocalFeatureVec HOG::GetFeature(const Mat& sketchImage) const
        {
            int orientNum = 4, sampleNum = 28, blockSize = 92, cellNum = 4;

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

            LocalFeatureVec feature;
            vector<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor descriptor = GetDescriptor(filteredOrientChannels, center, 
                    blockSize, cellNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline Descriptor HOG::GetDescriptor(const vector<Mat>& filteredOrientChannels, 
            const Point& center, int blockSize, int cellNum)
        {
            int height = filteredOrientChannels[0].rows, 
                width = filteredOrientChannels[0].cols;
            double cellSize = (double)blockSize / cellNum;
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
                        descriptor.push_back(hist.at<double>(i, j, k));

            NormTwoNormalize(descriptor.begin(), descriptor.end());
            return descriptor;
        }
        
        ///////////////////////////////////////////////////////////////////////

        class HOOSC : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "hoosc"; };

        private:
            static Descriptor GetDescriptor(const Mat& orientImage,
                const Point& pivot, const vector<Point>& points,
                const vector<double>& logDistances, int angleNum, int orientNum);
        };

        inline LocalFeatureVec HOOSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 9, orientNum = 8;

            vector<Point> points = GetEdgels(sketchImage);
            vector<Point> pivots = SampleFromPoints(points, (size_t)(points.size() * 0.33));

            tuple<Mat, Mat> gradient = Gradient::GetGradient(sketchImage);
            Mat& powerImage = get<0>(gradient);
            Mat& orientImage = get<1>(gradient);

            LocalFeatureVec feature;
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

            double orientStep = CV_PI / orientNum, sigma = 10;
	        for (int i = 0; i < pointNum; i++)
	        {
		        if (points[i] == pivot)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = FindBinIndex(angles[i], 0, 2 * CV_PI, angleNum, true);

                        double orient = orientImage.at<double>(points[i].y, points[i].x);
                        int o = (int)(orient / orientStep);

				        double value = Math::Gauss(((o + 0.5) * orientStep - orient) * 180 / CV_PI, sigma);
				        bins.at<double>(j, a, RoundIndex(o, orientNum, true)) += value;

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

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class SC : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "sc"; };

        private:
            static Descriptor GetDescriptor(const Point& pivot, const vector<Point>& pivots,
                const vector<double>& logDistances, int angleNum);
        };

        inline LocalFeatureVec SC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 12;

            vector<Point> points = GetEdgels(sketchImage);
            vector<Point> pivots = SampleFromPoints(points, (int)(points.size() * 0.33));

            LocalFeatureVec feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                Descriptor descriptor = GetDescriptor(pivots[i], pivots, 
                    logDistances, angleNum);
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

	        for (int i = 0; i < pivotNum; i++)
	        {
		        if (pivots[i] == pivot)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = FindBinIndex(angles[i], 0, 2 * CV_PI, angleNum, true);
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

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item);
	        }

	        return descriptor;
        }
        
        ///////////////////////////////////////////////////////////////////////

        class SHOG : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "shog"; };

        private:
            static Descriptor GetDescriptor(const vector<Mat>& orientChannels,
                const Point& pivot, const vector<Point>& points, int cellNum);
        };

        inline LocalFeatureVec SHOG::GetFeature(const Mat& sketchImage) const
        {
            int orientNum = 4, cellNum = 4;

            vector<Point> points = GetEdgels(sketchImage);
            vector<Point> pivots = SampleFromPoints(points, (int)(points.size() * 0.33));
            vector<Mat> orientChannels = Gradient::GetOrientChannels(sketchImage, orientNum);

            LocalFeatureVec feature;
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
            int blockSize = (int)(1.5 * mean);

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

                        int r = (int)((i - expectedTop) / cellSize), 
                            c = (int)((j - expectedLeft) / cellSize);

                        for (int u = -1; u <= 1; u++)
                        {
                            for (int v = -1; v <= 1; v++)
                            {
                                int newR = r + u, newC = c + v;
                                if (newR < 0 || newR >= cellNum || newC < 0 || newC >= cellNum)
                                    continue;

                                double dRatio = 1 - abs(Geometry::EulerDistance(
                                    Point((int)((newC + 0.5) * cellSize), (int)((newR + 0.5) * cellSize)),
                                    Point(j - expectedLeft, i - expectedTop))) / cellSize;
                                if (dRatio < 0)
                                    dRatio = 0;

                                hist.at<double>(newR, newC, k) += orientChannels[k].at<double>(i, j) * dRatio;
                            }
                        }
                    }                    
                }
            }

            Descriptor descriptor;
            for (int i = 0; i < cellNum; i++)
                for (int j = 0; j < cellNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        descriptor.push_back(hist.at<double>(i, j, k));

            NormTwoNormalize(descriptor.begin(), descriptor.end());
            return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class RHOOSC : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;
            
            virtual String GetName() const { return "rhoosc"; };

        private:
            static Descriptor GetDescriptor(const Mat& orientImage, 
                const Point& centre, const vector<Point>& points, 
                const vector<double>& logDistances, int angleNum, int orientNum);
        };

        inline LocalFeatureVec RHOOSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 9, orientNum = 8, sampleNum = 28;

            Mat orientImage = get<1>(Gradient::GetGradient(sketchImage));
            vector<Point> points = GetEdgels(sketchImage); 

            LocalFeatureVec feature;
            vector<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor desc = GetDescriptor(orientImage, center, points,
                    logDistances, angleNum, orientNum);
                feature.push_back(desc);
            }

            return feature;
        }

        inline Descriptor RHOOSC::GetDescriptor(const Mat& orientImage, 
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

            double orientStep = CV_PI / orientNum, sigma = 10;
	        for (int i = 0; i < pointNum; i++)
	        {
		        if (points[i] == centre)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = FindBinIndex(angles[i], 0, 2 * CV_PI, angleNum, true);

                        double orient = orientImage.at<double>(points[i].y, points[i].x);
                        int o = (int)(orient / orientStep); 

				        double value = Math::Gauss(((o + 0.5) * orientStep - orient) * 180 / CV_PI, sigma);
				        bins.at<double>(j, a, RoundIndex(o, orientNum, true)) += value;

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

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class RSC : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "rsc"; };
            
        private:
            static Descriptor GetDescriptor(const Point& center, 
                const vector<Point>& pivots, const vector<double>& logDistances, int angleNum);
        };

        inline LocalFeatureVec RSC::GetFeature(const Mat& sketchImage) const
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            vector<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 12, sampleNum = 28;

            Mat orientImage = get<1>(Gradient::GetGradient(sketchImage));
            vector<Point> points = GetEdgels(sketchImage); 
            vector<Point> pivots = SampleFromPoints(points, (int)(points.size() * 0.33));

            LocalFeatureVec feature;
            vector<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor desc = GetDescriptor(center, pivots, logDistances, angleNum);
                    feature.push_back(desc);
            }

            return feature;
        }

        inline Descriptor RSC::GetDescriptor(const Point& center, 
            const vector<Point>& pivots, const vector<double>& logDistances, int angleNum)
        {
            int pivotNum = pivots.size();
            assert(pivotNum > 1);

            vector<double> distances = Geometry::EulerDistance(center, pivots);
            vector<double> angles = Geometry::Angle(center, pivots);
	        double mean = Math::Sum(distances) / (pivotNum - 1); // Except pivot
	        for (int i = 0; i < pivotNum; i++)
		        distances[i] /= mean;

            int distanceNum = logDistances.size() - 1;
	        Mat bins = Mat::zeros(distanceNum, angleNum, CV_64F);

            double sigma = 10;
	        for (int i = 0; i < pivotNum; i++)
	        {
		        if (pivots[i] == center)
			        continue;

		        for (int j = 0; j < distanceNum; j++)
		        {
			        if (distances[i] >= logDistances[j] && distances[i] < logDistances[j + 1])
			        {
				        int a = FindBinIndex(angles[i], 0, 2 * CV_PI, angleNum, true);
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

		        NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.push_back(item);
	        }

	        return descriptor;
        }

        ///////////////////////////////////////////////////////////////////////

        class ASHOG : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "ashog"; };
            
        private:
            static Descriptor GetDescriptor(const vector<Mat>& orientChannels, 
                const Point& pivot, int blockSize, int cellNum);
        };

        inline LocalFeatureVec ASHOG::GetFeature(const Mat& sketchImage) const
        {
            int orientNum = 4, cellNum = 4, scaleNum = 15;
            double sigmaInit = 0.7, sigmaStep = 1.2;
            
            vector<double> sigmas;
            sigmas.push_back(sigmaInit);
            for (int i = 1; i < scaleNum; i++)
                sigmas.push_back(sigmas[i - 1] * sigmaStep);

            vector<Point> points = GetEdgels(sketchImage);
            vector<Point> pivots = SampleFromPoints(points, (int)(points.size() * 0.33));
            vector<Mat> orientChannels = Gradient::GetOrientChannels(sketchImage, orientNum);
            vector<Mat> pyramid = GetLoGPyramid(sketchImage, sigmas);
            
            LocalFeatureVec feature;
            for (int i = 0; i < pivots.size(); i++)
            {
                for (int j = 0; j < scaleNum; j++)
                {
                    double prev = j > 0 ? pyramid[j - 1].at<double>(pivots[i].y, pivots[i].x) : 0;
                    double curr = pyramid[j].at<double>(pivots[i].y, pivots[i].x);
                    double next = j < scaleNum - 1 ? pyramid[j + 1].at<double>(pivots[i].y, pivots[i].x) : 0;

                    if (curr > next && curr > prev)
                    {
                        Descriptor desc = GetDescriptor(orientChannels, pivots[i], 
                            (int)(sigmas[j] * 6 + 1), cellNum);
                        feature.push_back(desc);
                    }
                }
            }

            return feature;
        }

        inline Descriptor ASHOG::GetDescriptor(const vector<Mat>& orientChannels, 
            const Point& pivot, int blockSize, int cellNum)
        {
            int height = orientChannels[0].rows, 
                width = orientChannels[0].cols,
                orientNum = (int)orientChannels.size();
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

                        int r = (int)((i - expectedTop) / cellSize), 
                            c = (int)((j - expectedLeft) / cellSize);

                        for (int u = -1; u <= 1; u++)
                        {
                            for (int v = -1; v <= 1; v++)
                            {
                                int newR = r + u, newC = c + v;
                                if (newR < 0 || newR >= cellNum || newC < 0 || newC >= cellNum)
                                    continue;

                                double dRatio = 1 - abs(Geometry::EulerDistance(
                                    Point((int)((newC + 0.5) * cellSize), (int)((newR + 0.5) * cellSize)),
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

            NormTwoNormalize(desc.begin(), desc.end());
            return desc;
        }

        ///////////////////////////////////////////////////////////////////////

        class Gabor : public LocalFeature
        {
        protected:
            virtual LocalFeatureVec GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "gabor"; };
            
        private:
            static Descriptor GetDescriptor(const vector<Mat>& gaborResponses, 
                const Point& centre, int blockSize, int cellNum);
        };

        inline LocalFeatureVec Gabor::GetFeature(const Mat& sketchImage) const
        {
            int sampleNum = 28, blockSize = 92, cellNum = 4;
            int tmp[] = { 8, 8, 8, 8 };
            vector<int> orientNumPerScale(tmp, tmp + sizeof(tmp) / sizeof(int));

            vector<Mat> gaborResponses;
            for (int i = 0; i < orientNumPerScale.size(); i++)
            {
                double sigma = pow(1.8, i + 1), lambda = sigma * 1.7;

                int ksize = (int)(sigma * 6 + 1);
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

            LocalFeatureVec feature;
            vector<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor descriptor = GetDescriptor(gaborResponses, center, blockSize, cellNum);
                feature.push_back(descriptor);
            }

            return feature;
        }

        inline Descriptor Gabor::GetDescriptor(const vector<Mat>& gaborResponses, 
                const Point& centre, int blockSize, int cellNum)
        {
            assert(gaborResponses.size() > 0);

            int height = gaborResponses[0].rows,
                width = gaborResponses[0].cols,
                expectedTop = centre.y - blockSize / 2,
                expectedLeft = centre.x - blockSize / 2,
                cellSize = blockSize / cellNum;

            Descriptor desc;
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

        ///////////////////////////////////////////////////////////////////////

        class GIST : public GlobalFeature
        {
        protected:
            virtual GlobalFeatureVec GetFeature(const Mat& sketchImage) const;

            virtual String GetName() const { return "gist"; };

        private:
            static vector<Mat> GetGaborsInFreqDomain(const Size& size, 
                const vector<int>& orientNumPerScale);
        };

        inline GlobalFeatureVec GIST::GetFeature(const Mat& sketchImage) const
        {
            int blockNum = 4;
            int tmp[] = { 8, 8, 8, 8 };
            vector<int> orientNumPerScale(tmp, tmp + sizeof(tmp) / sizeof(int));

            vector<Mat> gaborsInFreqDomain = GetGaborsInFreqDomain(sketchImage.size(), 
                orientNumPerScale);
            
            Mat dftInReal, dftOutComplex, dftOutPlanes[2];
	        sketchImage.convertTo(dftInReal, CV_64FC1);
	        dft(dftInReal, dftOutComplex, DFT_COMPLEX_OUTPUT);
	        split(dftOutComplex, dftOutPlanes);

            GlobalFeatureVec feature;
	        for (int i = 0; i < gaborsInFreqDomain.size(); i++)
	        {
		        Mat idftInPlanes[] = { Mat::zeros(sketchImage.size(), CV_64F), 
                    Mat::zeros(sketchImage.size(), CV_64F) };
		        for (int j = 0; j < sketchImage.rows; j++)
			        for (int k = 0; k < sketchImage.cols; k++)
			        {
				        idftInPlanes[0].at<double>(j, k) = dftOutPlanes[0].at<double>(j, k) *
					        gaborsInFreqDomain[i].at<double>(j, k);
				        idftInPlanes[1].at<double>(j, k) = dftOutPlanes[1].at<double>(j, k) *
					        gaborsInFreqDomain[i].at<double>(j, k);
			        }

		        Mat idftInComplex, idftOutComplex, idftOutPlanes[2];
		        merge(idftInPlanes, 2, idftInComplex);
		        idft(idftInComplex, idftOutComplex, DFT_SCALE);
		        split(idftOutComplex, idftOutPlanes);

		        Mat finalImage;
		        magnitude(idftOutPlanes[0], idftOutPlanes[1], finalImage);

                int blockHeight = finalImage.rows / blockNum, 
                    blockWidth = finalImage.cols / blockNum;
		        for (int j = 0; j < blockNum; j++)
                {
                    for (int k = 0; k < blockNum; k++)
                    {
                        double sum = 0;
			            for (int r = 0; r < blockHeight; r++)
				            for (int c = 0; c < blockWidth; c++)
					            sum += finalImage.at<double>(j * blockHeight + r, k * blockWidth + c);

                        feature.push_back(sum / (blockWidth * blockHeight));
                    }
                }
	        }

            return feature;
        }
    
        inline vector<Mat> GIST::GetGaborsInFreqDomain(const Size& size, 
            const vector<int>& orientNumPerScale)
        {
            int height = size.height, width = size.width;

            int filterNum = 0;
	        for (int i = orientNumPerScale.size() - 1; i >= 0; i--)
		        filterNum += orientNumPerScale[i];

	        Mat param(filterNum, 4, CV_64F);
	        int l = 0;
	        for (int i = 0; i < orientNumPerScale.size(); i++)
            {
		        for (int j = 0; j < orientNumPerScale[i]; j++)
		        {
			        param.at<double>(l, 0) = 0.35;
			        param.at<double>(l, 1) = 0.3 / pow(1.85, i);
			        param.at<double>(l, 2) = 16 * pow(orientNumPerScale[i], 2) / pow(32, 2);
			        param.at<double>(l, 3) = CV_PI / orientNumPerScale[i] * j;
			        l++;
		        }
            }

	        Mat fp(size, CV_64F);
	        Mat fo(size, CV_64F);
	        for (int i = 0; i < height; i++)
            {
		        for (int j = 0; j < width; j++)
		        {
			        double fx = j - width / 2.0, fy = i - height / 2.0;
			        fp.at<double>(i, j) = sqrt(fx * fx + fy * fy);
			        fo.at<double>(i, j) = atan2(fy, fx);
		        }
            }
            fp = FFTShift(fp);
            fo = FFTShift(fo);

            vector<Mat> gaborsInFreqDomain;
	        for (int i = 0; i < filterNum; i++)
	        {
		        Mat gaborInFreqDomain(size, CV_64F);

		        for (int j = 0; j < height; j++)
			        for (int k = 0; k < width; k++)
			        {
				        double tmp = fo.at<double>(j, k) + param.at<double>(i, 3);
				        while (tmp < -CV_PI)
					        tmp += 2 * CV_PI;
				        while (tmp > CV_PI)
					        tmp -= 2 * CV_PI;

				        gaborInFreqDomain.at<double>(j, k) = exp(-10.0 * param.at<double>(i, 0) * 
					        (fp.at<double>(j, k) / height / param.at<double>(i, 1) - 1) * 
					        (fp.at<double>(j, k) / width / param.at<double>(i, 1) - 1) - 
					        2.0 * param.at<double>(i, 2) * CV_PI * tmp * tmp);

			        }

		        gaborsInFreqDomain.push_back(gaborInFreqDomain);
	        }

            return gaborsInFreqDomain;
        }
    }
}