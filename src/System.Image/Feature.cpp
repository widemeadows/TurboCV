#include "../System/System.h"
#include "../System.ML/System.ML.h"
#include "Core.h"
#include "Recognition.h"
#include <cv.h>
#include <highgui.h>
#include <cstdio>
#include <functional>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // Save Features
        //////////////////////////////////////////////////////////////////////////

        void SaveLocalFeatures(
            const TString& fileName, 
            const ArrayList<double>& sigmas,
            const ArrayList<Word_f>& words,
            const ArrayList<Histogram>& freqHists, 
            const ArrayList<int>& labels)
        {
            if (words.Count() == 0 || freqHists.Count() == 0)
                return;

            FILE* file = fopen(fileName, "w");

            int nWord = words.Count();      
            int nDim = words[0].Count();

            fprintf(file, "%d\n", nWord);
            for (int i = 0; i < nWord; i++)
                fprintf(file, "%f ", sigmas[i]);
            fprintf(file, "\n");

            fprintf(file, "%d %d\n", nWord, nDim);
            for (int i = 0; i < nWord; i++)
            {
                for (int j = 0; j < nDim; j++)
                    fprintf(file, "%f ", words[i][j]);
                fprintf(file, "\n");
            }

            int nHist = freqHists.Count();
            nDim = freqHists[0].Count();

            fprintf(file, "%d %d\n", nHist, nDim);
            for (int i = 0; i < nHist; i++)
            {
                fprintf(file, "%d ", labels[i]);
                for (int j = 0; j < nDim; j++)
                    fprintf(file, "%f ", freqHists[i][j]);
                fprintf(file, "\n");
            }

            fclose(file);
        }

        void SaveLocalFeatures(
            const TString& fileName, 
            const ArrayList<Word_f>& words,
            const ArrayList<Histogram>& freqHists, 
            const ArrayList<int>& labels)
        {
            if (words.Count() == 0 || freqHists.Count() == 0)
                return;

            FILE* file = fopen(fileName, "w");

            int nWord = words.Count();      
            int nDim = words[0].Count();

            fprintf(file, "%d %d\n", nWord, nDim);
            for (int i = 0; i < nWord; i++)
            {
                for (int j = 0; j < nDim; j++)
                    fprintf(file, "%f ", words[i][j]);
                fprintf(file, "\n");
            }

            int nHist = freqHists.Count();
            nDim = freqHists[0].Count();

            fprintf(file, "%d %d\n", nHist, nDim);
            for (int i = 0; i < nHist; i++)
            {
                fprintf(file, "%d ", labels[i]);
                for (int j = 0; j < nDim; j++)
                    fprintf(file, "%f ", freqHists[i][j]);
                fprintf(file, "\n");
            }

            fclose(file);
        }

        void SaveGlobalFeatures(
            const TString& fileName, 
            const ArrayList<GlobalFeatureVec_f>& features,
            const ArrayList<int>& labels)
        {
            if (features.Count() == 0)
                return;

            FILE* file = fopen(fileName, "w");
            int nFeature = features.Count();
            int nDim = features[0].Count();

            fprintf(file, "%d %d\n", nFeature, nDim);

            for (int i = 0; i < nFeature; i++)
            {
                fprintf(file, "%d ", labels[i]);
                for (int j = 0; j < nDim; j++)
                    fprintf(file, "%f ", features[i][j]);
                fprintf(file, "\n");
            }

            fclose(file);
        }

        void SaveDistanceMatrix(
            const TString& fileName,
            const cv::Mat& distanceMatrix,
            const ArrayList<int>& labels
            )
        {
            FILE* file = fopen(fileName, "w");

            fprintf(file, "%d\n", distanceMatrix.rows);
            for (int i = 0; i < distanceMatrix.rows; i++)
            {
                fprintf(file, "%d", labels[i]);
                for (int j = 0; j < distanceMatrix.cols; j++)
                    fprintf(file, " %f", distanceMatrix.at<double>(i, j));
                fprintf(file, "\n");
            }

            fclose(file);
        }


        //////////////////////////////////////////////////////////////////////////
        // HOG
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec HOG::GetFeature(const Mat& sketchImage)
        {
            int kernelSize = cellSize * 2 + 1;
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

            ArrayList<Mat> orientChannels = GetOrientChannels(sketchImage, orientNum);
            ArrayList<Mat> filteredOrientChannels(orientNum);
            for (int i = 0; i < orientNum; i++)
                filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

            LocalFeatureVec feature;
            for (int i = 0; i < sketchImage.rows; i += cellSize)
            {
                for (int j = 0; j < sketchImage.cols; j += cellSize)
                {
                    Descriptor descriptor = GetDescriptor(filteredOrientChannels, Point(j, i));
                    feature.Add(descriptor);
                }
            }

            return feature;
        }

        Descriptor HOG::GetDescriptor(const ArrayList<Mat>& filteredOrientChannels, 
            const Point& center)
        {
            int height = filteredOrientChannels[0].rows, 
                width = filteredOrientChannels[0].cols;
            int blockSize = cellSize * cellNum;
            int expectedTop = (int)(center.y - blockSize / 2),
                expectedLeft = (int)(center.x - blockSize / 2);
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


        //////////////////////////////////////////////////////////////////////////
        // RHOG
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec RHOG::GetFeature(const Mat& sketchImage)
        {
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

            ArrayList<Mat> orientChannels = GetOrientChannels(sketchImage, orientNum);
            ArrayList<Mat> filteredOrientChannels(orientNum);
            for (int i = 0; i < orientNum; i++)
                filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

            LocalFeatureVec feature;
            ArrayList<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor descriptor = GetDescriptor(filteredOrientChannels, center);
                feature.Add(descriptor);
            }

            return feature;
        }

        Descriptor RHOG::GetDescriptor(const ArrayList<Mat>& filteredOrientChannels, 
            const Point& center)
        {
            int height = filteredOrientChannels[0].rows, 
                width = filteredOrientChannels[0].cols;
            double cellSize = blockSize / cellNum;
            int expectedTop = center.y - blockSize / 2,
                expectedLeft = center.x - blockSize / 2;
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


        //////////////////////////////////////////////////////////////////////////
        // SHOG
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec SHOG::GetFeature(const Mat& sketchImage)
        {
            ArrayList<Mat> orientChannels = GetOrientChannels(sketchImage, orientNum);
            ArrayList<Point> points = GetEdgels(sketchImage);
            ArrayList<Point> pivots = SampleFromPoints(points, (size_t)(points.Count() * pivotRatio + 0.5));

            LocalFeatureVec feature;
            for (int i = 0; i < pivots.Count(); i++)
            {
                Descriptor descriptor = GetDescriptor(orientChannels, pivots[i], points);
                feature.Add(descriptor);
            }

            return feature;
        }

        Descriptor SHOG::GetDescriptor(const ArrayList<Mat>& orientChannels,
            const Point& pivot, const ArrayList<Point>& points)
        {
            ArrayList<double> distances = EulerDistance(pivot, points);
            double mean = Math::Sum(distances) / (points.Count() - 1); // Except pivot
            double blockSize = 1.5 * mean;

            int height = orientChannels[0].rows, 
                width = orientChannels[0].cols;
            int expectedTop = pivot.y - blockSize / 2,
                expectedLeft = pivot.x - blockSize / 2;
            double cellSize = blockSize / cellNum;
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

                                double dRatio = 1 - abs(EulerDistance(
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
                        descriptor.Add(hist.at<double>(i, j, k));

            NormTwoNormalize(descriptor.begin(), descriptor.end());
            return descriptor;
        }


        //////////////////////////////////////////////////////////////////////////
        // LogSHOG
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec LogSHOG::GetFeature(const Mat& sketchImage)
        {
            ArrayList<double> sigmas;
            sigmas.Add(sigmaInit);
            for (int i = 1; i < scaleNum; i++)
                sigmas.Add(sigmas[i - 1] * sigmaStep);

            ArrayList<Point> points = GetEdgels(sketchImage);
            ArrayList<Point> pivots = SampleFromPoints(points, (size_t)(points.Count() * pivotRatio + 0.5));
            ArrayList<Mat> orientChannels = GetOrientChannels(sketchImage, orientNum);
            ArrayList<Mat> pyramid = GetLoGPyramid(sketchImage, sigmas);

            LocalFeatureVec feature;
            for (int i = 0; i < pivots.Count(); i++)
            {
                for (int j = 0; j < scaleNum; j++)
                {
                    double prev = j > 0 ? pyramid[j - 1].at<double>(pivots[i].y, pivots[i].x) : 0;
                    double curr = pyramid[j].at<double>(pivots[i].y, pivots[i].x);
                    double next = j < scaleNum - 1 ? pyramid[j + 1].at<double>(pivots[i].y, pivots[i].x) : 0;

                    if (curr > next && curr > prev)
                    {
                        Descriptor desc = GetDescriptor(orientChannels, pivots[i], sigmas[j] * 6);
                        feature.Add(desc);
                    }
                }
            }

            return feature;
        }

        Descriptor LogSHOG::GetDescriptor(const ArrayList<Mat>& orientChannels, 
            const Point& pivot, double blockSize)
        {
            int height = orientChannels[0].rows, 
                width = orientChannels[0].cols;
            int expectedTop = pivot.y - blockSize / 2,
                expectedLeft = pivot.x - blockSize / 2;
            double cellSize = blockSize / cellNum;
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

                                double dRatio = 1 - abs(EulerDistance(
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
                        desc.Add(hist.at<double>(i, j, k));

            NormTwoNormalize(desc.begin(), desc.end());
            return desc;
        }


        //////////////////////////////////////////////////////////////////////////
        // HOOSC
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec HOOSC::GetFeature(const Mat& sketchImage)
        {
            ArrayList<Point> points = GetEdgels(sketchImage);
            ArrayList<Point> pivots = SampleFromPoints(points, (size_t)(points.Count() * pivotRatio + 0.5));

            Group<Mat, Mat> gradient = GetGradient(sketchImage);
            Mat& powerImage = gradient.Item1();
            Mat& orientImage = gradient.Item2();

            LocalFeatureVec feature;
            for (int i = 0; i < pivots.Count(); i++)
            {
                Descriptor descriptor = GetDescriptor(orientImage, pivots[i], points);
                feature.Add(descriptor);
            }

            return feature;
        }

        Descriptor HOOSC::GetDescriptor(const Mat& orientImage, const Point& pivot, 
            const ArrayList<Point>& points)
        {
            int pointNum = points.Count();
            assert(pointNum > 1);

            ArrayList<double> distances = EulerDistance(pivot, points);
            ArrayList<double> angles = Angle(pivot, points);

            double mean = Math::Sum(distances) / (pointNum - 1); // Except pivot
            for (int i = 0; i < pointNum; i++)
                distances[i] /= mean;

            int distanceNum = logDistances.Count() - 1;
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
                ArrayList<double> ring;
                for (int j = 0; j < angleNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        ring.Add(bins.at<double>(i, j, k));

                NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.Add(item);
            }

            return descriptor;
        }


        //////////////////////////////////////////////////////////////////////////
        // RHOOSC
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec RHOOSC::GetFeature(const Mat& sketchImage)
        {
            Mat orientImage = GetGradient(sketchImage).Item2();
            ArrayList<Point> points = GetEdgels(sketchImage); 

            LocalFeatureVec feature;
            ArrayList<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor desc = GetDescriptor(orientImage, center, points);
                feature.Add(desc);
            }

            return feature;
        }

        Descriptor RHOOSC::GetDescriptor(const Mat& orientImage, const Point& center, 
            const ArrayList<Point>& points)
        {
            int pointNum = points.Count();
            assert(pointNum > 1);

            ArrayList<double> distances = EulerDistance(center, points);
            ArrayList<double> angles = Angle(center, points);

            double mean;
            if (points.Contains(center))
                mean = Math::Sum(distances) / (pointNum - 1); // Except center
            else
                mean = Math::Sum(distances) / pointNum;

            for (int i = 0; i < pointNum; i++)
                distances[i] /= mean;

            int distanceNum = logDistances.Count() - 1;
            int dims[] = { distanceNum, angleNum, orientNum };
            Mat bins(3, dims, CV_64F);
            bins = Scalar::all(0);
            double orientStep = CV_PI / orientNum, sigma = 10;
            for (int i = 0; i < pointNum; i++)
            {
                if (points[i] == center)
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
                ArrayList<double> ring;
                for (int j = 0; j < angleNum; j++)
                    for (int k = 0; k < orientNum; k++)
                        ring.Add(bins.at<double>(i, j, k));

                NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.Add(item);
            }

            return descriptor;
        }


        //////////////////////////////////////////////////////////////////////////
        // SC
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec SC::GetFeature(const Mat& sketchImage)
        {
            ArrayList<Point> points = GetEdgels(sketchImage);
            ArrayList<Point> pivots = SampleFromPoints(points, (size_t)(points.Count() * pivotRatio + 0.5));

            LocalFeatureVec feature;
            for (int i = 0; i < pivots.Count(); i++)
            {
                Descriptor descriptor = GetDescriptor(pivots[i], pivots);
                feature.Add(descriptor);
            }

            return feature;
        }

        Descriptor SC::GetDescriptor(const Point& pivot, const ArrayList<Point>& pivots)
        {
            int pivotNum = pivots.Count();
            assert(pivotNum > 1);

            ArrayList<double> distances = EulerDistance(pivot, pivots);
            ArrayList<double> angles = Angle(pivot, pivots);
            double mean = Math::Sum(distances) / (pivotNum - 1); // Except pivot
            for (int i = 0; i < pivotNum; i++)
                distances[i] /= mean;

            int distanceNum = logDistances.Count() - 1;
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
                ArrayList<double> ring;
                for (int j = 0; j < angleNum; j++)
                    ring.Add(bins.at<double>(i, j));

                NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.Add(item);
            }

            return descriptor;
        }


        //////////////////////////////////////////////////////////////////////////
        // PSC
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec PSC::GetFeature(const Mat& sketchImage)
        {
            ArrayList<Point> points = GetEdgels(sketchImage);
            ArrayList<Point> pivots = SampleFromPoints(points, (size_t)(points.Count() * pivotRatio + 0.5));

            LocalFeatureVec feature;
            for (int i = 0; i < pivots.Count(); i++)
            {
                Descriptor descriptor = GetDescriptor(pivots[i], points);
                feature.Add(descriptor);
            }

            return feature;
        }

        Descriptor PSC::GetDescriptor(const Point& pivot, const ArrayList<Point>& points)
        {
            int pointNum = points.Count();
            assert(pointNum > 1);

            ArrayList<double> distances = EulerDistance(pivot, points);
            ArrayList<double> angles = Angle(pivot, points);
            double mean = Math::Sum(distances) / (pointNum - 1);
            for (int i = 0; i < pointNum; i++)
                distances[i] /= mean;

            int distanceNum = logDistances.Count() - 1;
            Mat bins = Mat::zeros(distanceNum, angleNum, CV_64F);
            for (int i = 0; i < pointNum; i++)
            {
                if (points[i] == pivot)
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
                ArrayList<double> ring;
                for (int j = 0; j < angleNum; j++)
                    ring.Add(bins.at<double>(i, j));

                NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.Add(item);
            }

            return descriptor;
        }


        //////////////////////////////////////////////////////////////////////////
        // RSC
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec RSC::GetFeature(const Mat& sketchImage)
        {
            double tmp[] = { 0, 0.125, 0.25, 0.5, 1, 2 };
            ArrayList<double> logDistances(tmp, tmp + sizeof(tmp) / sizeof(double));
            int angleNum = 12, sampleNum = 28;

            Mat orientImage = GetGradient(sketchImage).Item2();
            ArrayList<Point> points = GetEdgels(sketchImage); 
            ArrayList<Point> pivots = SampleFromPoints(points, (size_t)(points.Count() * pivotRatio + 0.5));

            LocalFeatureVec feature;
            ArrayList<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor desc = GetDescriptor(center, pivots);
                feature.Add(desc);
            }

            return feature;
        }

        Descriptor RSC::GetDescriptor(const Point& center, const ArrayList<Point>& pivots)
        {
            int pivotNum = pivots.Count();
            assert(pivotNum > 1);

            ArrayList<double> distances = EulerDistance(center, pivots);
            ArrayList<double> angles = Angle(center, pivots);

            double mean;
            if (pivots.Contains(center))
                mean = Math::Sum(distances) / (pivotNum - 1); // Except pivot
            else
                mean = Math::Sum(distances) / pivotNum;

            for (int i = 0; i < pivotNum; i++)
                distances[i] /= mean;

            int distanceNum = logDistances.Count() - 1;
            Mat bins = Mat::zeros(distanceNum, angleNum, CV_64F);
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
                ArrayList<double> ring;
                for (int j = 0; j < angleNum; j++)
                    ring.Add(bins.at<double>(i, j));

                NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.Add(item);
            }

            return descriptor;
        }


        //////////////////////////////////////////////////////////////////////////
        // RPSC
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec RPSC::GetFeature(const Mat& sketchImage)
        {
            Mat orientImage = GetGradient(sketchImage).Item2();
            ArrayList<Point> points = GetEdgels(sketchImage); 

            LocalFeatureVec feature;
            ArrayList<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor desc = GetDescriptor(center, points, logDistances, angleNum);
                feature.Add(desc);
            }

            return feature;
        }

        Descriptor RPSC::GetDescriptor(const Point& center, const ArrayList<Point>& points, 
            const ArrayList<double>& logDistances, int angleNum)
        {
            int pointNum = points.Count();
            assert(pointNum > 1);

            ArrayList<double> distances = EulerDistance(center, points);
            ArrayList<double> angles = Angle(center, points);

            double mean;
            if (points.Contains(center))
                mean = Math::Sum(distances) / (pointNum - 1); // Except pivot
            else
                mean = Math::Sum(distances) / pointNum;

            for (int i = 0; i < pointNum; i++)
                distances[i] /= mean;

            int distanceNum = logDistances.Count() - 1;
            Mat bins = Mat::zeros(distanceNum, angleNum, CV_64F);
            for (int i = 0; i < pointNum; i++)
            {
                if (points[i] == center)
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
                ArrayList<double> ring;
                for (int j = 0; j < angleNum; j++)
                    ring.Add(bins.at<double>(i, j));

                NormOneNormalize(ring.begin(), ring.end());

                for (auto item : ring)
                    descriptor.Add(item);
            }

            return descriptor;
        }


        //////////////////////////////////////////////////////////////////////////
        // RGabor
        //////////////////////////////////////////////////////////////////////////

        LocalFeatureVec RGabor::GetFeature(const Mat& sketchImage)
        {
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

            ArrayList<Mat> orientChannels = GetGaborChannels(sketchImage, orientNum);
            ArrayList<Mat> filteredOrientChannels(orientNum);
            for (int i = 0; i < orientNum; i++)
                filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

            LocalFeatureVec feature;
            ArrayList<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor descriptor = GetDescriptor(filteredOrientChannels, center);
                feature.Add(descriptor);
            }

            return feature;
        }

        Descriptor RGabor::GetDescriptor(const ArrayList<Mat>& filteredOrientChannels, 
            const Point& center)
        {
            int height = filteredOrientChannels[0].rows, 
                width = filteredOrientChannels[0].cols;
            double cellSize = blockSize / cellNum;
            int expectedTop = center.y - blockSize / 2,
                expectedLeft = center.x - blockSize / 2;
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

        //////////////////////////////////////////////////////////////////////////
        // GHOG
        //////////////////////////////////////////////////////////////////////////

        GlobalFeatureVec GHOG::GetFeature(const Mat& sketchImage)
        {
            int kernelSize = blockSize * 2 + 1;
            Mat tentKernel(kernelSize, kernelSize, CV_64F);
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    double ratio = 1 - sqrt((i - blockSize) * (i - blockSize) + 
                        (j - blockSize) * (j - blockSize)) / blockSize;
                    if (ratio < 0)
                        ratio = 0;

                    tentKernel.at<double>(i, j) = ratio;
                }
            }

            ArrayList<Mat> orientChannels = GetOrientChannels(sketchImage, orientNum);
            ArrayList<Mat> filteredOrientChannels(orientNum);
            for (int i = 0; i < orientNum; i++)
                filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

            GlobalFeatureVec feature;
            for (int i = blockSize / 2 - 1; i < sketchImage.rows; i += blockSize / 2)
                for (int j = blockSize / 2 - 1; j < sketchImage.cols; j += blockSize / 2)
                    for (int k = 0; k < orientNum; k++)
                        feature.Add(filteredOrientChannels[k].at<double>(i, j));

            NormTwoNormalize(feature.begin(), feature.end());
            return feature;
        }


        //////////////////////////////////////////////////////////////////////////
        // LGBPHS
        //////////////////////////////////////////////////////////////////////////

        GlobalFeatureVec LGBPHS::GetFeature(const Mat& sketchImage)
        {
            int blockSize = sketchImage.rows / blockNum;
            ArrayList<Mat> orientChannels = GetOrientChannels(sketchImage, orientNum);
            
            ArrayList<Mat> lbpChannels(orientNum);
            for (int i = 0; i < orientNum; i++)
            {
                Mat lbpMap(orientChannels[i].size(), CV_8U);

                for (int j = 0; j < orientChannels[i].rows; j++)
                for (int k = 0; k < orientChannels[i].cols; k++)
                {
                    int factor = 1;
                    uchar lbp = 0;

                    for (int m = -1; m <= 1; m++)
                    for (int n = -1; n <= 1; n++)
                    {
                        if (m == 0 && n == 0)
                            continue;

                        int newR = j + m, newC = k + n;
                        if (newR >= 0 && newR < orientChannels[i].rows &&
                            newC >= 0 && newC < orientChannels[i].cols &&
                            orientChannels[i].at<double>(newR, newC) > orientChannels[i].at<double>(j, k))
                            lbp += factor;

                        factor *= 2;
                    }

                    lbpMap.at<uchar>(j, k) = lbp;
                }

                lbpChannels[i] = lbpMap;
            }

            GlobalFeatureVec feature;
            for (int i = blockSize / 2; i < sketchImage.rows; i += blockSize)
            for (int j = blockSize / 2; j < sketchImage.cols; j += blockSize)
            for (int k = 0; k < orientNum; k++)
            {
                int top = max(0, i - blockSize / 2),
                    bottom = min(sketchImage.rows, i + blockSize / 2),
                    left = max(0, j - blockSize / 2),
                    right = min(sketchImage.cols, j + blockSize / 2);

                ArrayList<double> histogram(256);
                for (int m = top; m < bottom; m++)
                for (int n = left; n < right; n++)
                    histogram[lbpChannels[k].at<uchar>(m, n)]++;

                for (auto item : histogram)
                    feature.Add(item);
            }

            NormTwoNormalize(feature.begin(), feature.end());
            return feature;
        }


        //////////////////////////////////////////////////////////////////////////
        // GIST
        //////////////////////////////////////////////////////////////////////////

        GlobalFeatureVec GIST::GetFeature(const Mat& sketchImage)
        {
            ArrayList<Mat> gaborsInFreqDomain = GetGaborsInFreqDomain(sketchImage.size());

            Mat dftInReal, dftOutComplex, dftOutPlanes[2];
            sketchImage.convertTo(dftInReal, CV_64FC1);
            dft(dftInReal, dftOutComplex, DFT_COMPLEX_OUTPUT);
            split(dftOutComplex, dftOutPlanes);

            GlobalFeatureVec feature;
            for (int i = 0; i < gaborsInFreqDomain.Count(); i++)
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

                            feature.Add(sum / (blockWidth * blockHeight));
                        }
                    }
            }

            NormTwoNormalize(feature.begin(), feature.end());
            return feature;
        }

        ArrayList<Mat> GIST::GetGaborsInFreqDomain(const Size& size)
        {
            int height = size.height, width = size.width;
            int filterNum = Math::Sum(orientNumPerScale);

            Mat param(filterNum, 4, CV_64F);
            int l = 0;
            for (int i = 0; i < orientNumPerScale.Count(); i++)
            {
                for (int j = 0; j < orientNumPerScale[i]; j++)
                {
                    param.at<double>(l, 0) = 0.35;
                    param.at<double>(l, 1) = 0.3 / pow(1.85, i);
                    param.at<double>(l, 2) = 16.0 * pow(orientNumPerScale[i], 2) / pow(32, 2);
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

            ArrayList<Mat> gaborsInFreqDomain;
            for (int i = 0; i < filterNum; i++)
            {
                Mat gaborInFreqDomain(size, CV_64F);

                for (int j = 0; j < height; j++)
                {
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
                }

                gaborsInFreqDomain.Add(gaborInFreqDomain);
            }

            return gaborsInFreqDomain;
        }


        //////////////////////////////////////////////////////////////////////////
        // TGabor
        //////////////////////////////////////////////////////////////////////////

        ArrayList<Mat> GetChannels(const Mat& sketchImage, int orientNum, int angleNum)
        {
            int ksize = 10;

            ArrayList<PointList> tmp = GetEdgelChannels(sketchImage, orientNum);
            ArrayList<Mat> pChannels(orientNum);
            for (int i = 0; i < pChannels.Count(); i++)
            {
                Mat mat = Mat::zeros(sketchImage.size(), CV_8U);
                for (int j = 0; j < tmp[i].Count(); j++)
                    mat.at<uchar>(tmp[i][j].y, tmp[i][j].x) = 255;
                pChannels[i] = mat;
            }
            
            ArrayList<Mat> triChannels(angleNum * orientNum);
            for (int i = 0; i < triChannels.Count(); i++)
                triChannels[i] = Mat::zeros(sketchImage.size(), CV_64F);

            for (int c = 0; c < pChannels.Count(); c++)
            for (int i = 0; i < sketchImage.rows; i++)
            for (int j = 0; j < sketchImage.cols; j++)
            if (pChannels[c].at<uchar>(i, j))
            {
                int bottom = max(i - ksize, 0),
                    top = min(i + ksize, sketchImage.rows - 1),
                    left = max(j - ksize, 0),
                    right = min(j + ksize, sketchImage.cols - 1);

                Point u(j, i);

                ArrayList<Point> points;
                for (int m = bottom; m <= top; m++)
                for (int n = left; n <= right; n++)
                {
                    Point v(n, m);
                    if (u == v || !pChannels[c].at<uchar>(m, n))
                        continue;

                    points.Add(v);
                }

                sort(points.begin(), points.end(), [](const Point& u, const Point& v)
                {
                    if (u.x != v.x)
                        return u.x < v.x;
                    else
                        return u.y < v.y;
                });

                ArrayList<double> tmp(angleNum);
                for (int m = 0; m < points.Count(); m++)
                for (int n = m + 1; n < points.Count(); n++)
                {
                    Point mu = points[m] - u;
                    Point nu = points[n] - u;
                    Point mn = points[m] - points[n];
                    double cosian = (mu.ddot(mu) + nu.ddot(nu) - mn.ddot(mn)) / (2 * sqrt(mu.ddot(mu)) * sqrt(nu.ddot(nu)));
                    cosian = min(1.0, cosian);
                    cosian = max(-1.0, cosian);
                    double angle = acos(cosian);
                    int a = FindBinIndex(angle, 0, CV_PI, angleNum, true);
                    triChannels[c * angleNum + a].at<double>(i, j)++;
                }
            }

            return triChannels;

            /*ArrayList<Mat> channels(orientNum * angleNum);
            for (int i = 0; i < channels.Count(); i++)
                channels[i] = Mat::zeros(sketchImage.size(), CV_64F);
            
            ArrayList<Mat> gaborChannels = GetGaborChannels(sketchImage, orientNum);
            ArrayList<PointList> points = GetEdgelChannels(sketchImage, orientNum);
            for (int i = 0; i < orientNum; i++)
            {
                sort(points[i].begin(), points[i].end(), [](const Point& u, const Point& v)
                {
                    if (u.x != v.x)
                        return u.x < v.x;
                    else
                        return u.y < v.y;
                });

                for (int j = 0; j < points[i].Count(); j++)
                {
                    ArrayList<double> tmp(angleNum);

                    for (int k = 0; k < points[i].Count(); k++)
                    {
                        if (points[i][j] == points[i][k])
                            continue;

                        double angle = Angle(points[i][j], points[i][k]);
                        int a = FindBinIndex(angle, 0, 2 * CV_PI, angleNum, true);
                        tmp[a]++;
                    }

                    NormOneNormalize(tmp.begin(), tmp.end());
                    for (int k = 0; k < tmp.Count(); k++)
                        channels[i * angleNum + k].at<double>(points[i][j].y, points[i][j].x) = tmp[k];
                }
            }

            return channels;*/
        }

        LocalFeatureVec TGabor::GetFeature(const Mat& sketchImage)
        {
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

            ArrayList<Mat> orientChannels = GetChannels(sketchImage, orientNum, angleNum);
            ArrayList<Mat> filteredOrientChannels(orientChannels.Count());
            for (int i = 0; i < orientChannels.Count(); i++)
                filter2D(orientChannels[i], filteredOrientChannels[i], -1, tentKernel);

            LocalFeatureVec feature;
            ArrayList<Point> centers = SampleOnGrid(sketchImage.rows, sketchImage.cols, sampleNum);
            for (Point center : centers)
            {
                Descriptor descriptor = GetDescriptor(filteredOrientChannels, center);
                feature.Add(descriptor);
            }

            printf(".");
            return feature;
        }

        Descriptor TGabor::GetDescriptor(const ArrayList<Mat>& filteredOrientChannels,
            const Point& center)
        {
            int height = filteredOrientChannels[0].rows,
                width = filteredOrientChannels[0].cols;
            double cellSize = blockSize / cellNum;
            int expectedTop = center.y - blockSize / 2,
                expectedLeft = center.x - blockSize / 2;
            int dims[] = { cellNum, cellNum, filteredOrientChannels.Count() };
            Mat hist(3, dims, CV_64F);

            for (int i = 0; i < cellNum; i++)
            {
                for (int j = 0; j < cellNum; j++)
                {
                    for (int k = 0; k < filteredOrientChannels.Count(); k++)
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
            for (int k = 0; k < filteredOrientChannels.Count(); k++)
                descriptor.Add(hist.at<double>(i, j, k));

            NormTwoNormalize(descriptor.begin(), descriptor.end());
            return descriptor;
        }
        
        // GMMtSL
        GlobalFeatureVec GMMtSL::GetFeature(const Mat& sketchImage)
        {
            int blockNum = 2;
            int blockSize = sketchImage.rows / blockNum;
            GlobalFeatureVec feature;

            for (int i = 0; i < sketchImage.rows; i += blockSize)
            for (int j = 0; j < sketchImage.cols; j += blockSize)
            {
                Mat x = EdgelsToMat(GetEdgels(Mat(sketchImage, Rect(j, i, blockSize, blockSize))));
                ML::GMMFeature gmmFeature = ML::GMM(x, 5).Item2();

                feature.Add(gmmFeature.ToArrayList());
            }

            cout << ".";
            return feature;
        }
    }
}
}
