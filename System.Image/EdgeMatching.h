#pragma once

#include "../System/System.h"
#include "Feature.h"
#include "Filter.h"
#include <cv.h>
using namespace std;
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        // Chamfer Matching
        class CM : public Feature
        {
        public:
            typedef Tuple<vector<Point>, Mat> Info;

            Info GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false) const;
            Info GetFeatureWithoutPreprocess(const Mat& sketchImage) const;

            static double GetDistance(const Info& u, const Info& v);

            virtual String GetName() const { return "cm"; };

        protected:
            virtual Info Transform(const Mat& sketchImage, double maxDistance = 40) const;
        };

        inline CM::Info CM::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning) const
        {
            return Transform(Preprocess(sketchImage, thinning));
        }

        inline CM::Info CM::GetFeatureWithoutPreprocess(const Mat& sketchImage) const
        {
            return Transform(sketchImage);
        }

        inline double CM::GetDistance(const Info& u, const Info& v)
        {
            const vector<Point>& uPoints = u.Item1();
            const vector<Point>& vPoints = v.Item1();
            const Mat& uMat = u.Item2();
            const Mat& vMat = v.Item2();
            double uToV = 0, vToU = 0;

            for (size_t i = 0; i < uPoints.size(); i++)
                uToV += vMat.at<double>(uPoints[i].y, uPoints[i].x);

            for (size_t i = 0; i < vPoints.size(); i++)
                vToU += uMat.at<double>(vPoints[i].y, vPoints[i].x);

            if (uPoints.size() == 0 || vPoints.size() == 0)
                return 1;
            else
                return (uToV / uPoints.size() + vToU / vPoints.size()) / 2.0;
        }

        inline CM::Info CM::Transform(const Mat& sketchImage, double maxDistance) const
        {
            Mat dt(sketchImage.size(), CV_64F);

            for (int i = 0; i < dt.rows; i++)
                for (int j = 0; j < dt.cols; j++)
                    dt.at<double>(i, j) = maxDistance * maxDistance;

            vector<Point> points = GetEdgels(sketchImage);

            for (size_t i = 0; i < points.size(); i++)
            {
                int left = (int)floor(points[i].x - maxDistance),
                    right = (int)ceil(points[i].x + maxDistance),
                    top = (int)floor(points[i].y - maxDistance),
                    bottom = (int)ceil(points[i].y + maxDistance);
                left = left < 0 ? 0 : left;
                right = right > dt.cols ? dt.cols : right;
                top = top < 0 ? 0 : top;
                bottom = bottom > dt.rows ? dt.rows : bottom;

                for (int m = top; m < bottom; m++)
                {
                    for (int n = left; n < right; n++)
                    {
                        double distance = (m - points[i].y) * (m - points[i].y) + 
                            (n - points[i].x) * (n - points[i].x);

                        dt.at<double>(m, n) = min(distance, dt.at<double>(m, n));
                    }
                }
            }

            return CreateTuple(points, dt);
        }

        ///////////////////////////////////////////////////////////////////////

        // Oriented Chamfer Matching
        class OCM : public Feature
        {
        public:
            typedef Vector<Tuple<vector<Point>, Mat>> Info;

            Info GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false);
            Info GetFeatureWithoutPreprocess(const Mat& sketchImage);

            static double GetDistance(const Info& u, const Info& v);
            static vector<vector<Point>> GetChannels(const Mat& sketchImage, int orientNum);

            virtual String GetName() const { return "ocm"; };

        protected:
            virtual Info Transform(const Mat& sketchImage, double maxDistance = 40);
        };

        inline OCM::Info OCM::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning)
        {
            return Transform(Preprocess(sketchImage, thinning));
        }

        inline OCM::Info OCM::GetFeatureWithoutPreprocess(const Mat& sketchImage)
        {
            return Transform(sketchImage);
        }

        inline double OCM::GetDistance(const Info& u, const Info& v)
        {
            assert(u.size() == v.size());
            int orientNum = u.size(), uPointNum = 0, vPointNum = 0;
            double uToV = 0, vToU = 0;

            for (int i = 0; i < orientNum; i++)
            {
                const vector<Point>& uPoints = u[i].Item1();
                const vector<Point>& vPoints = v[i].Item1();
                const Mat& uMat = u[i].Item2();
                const Mat& vMat = v[i].Item2();

                for (size_t i = 0; i < uPoints.size(); i++)
                    uToV += vMat.at<float>(uPoints[i].y, uPoints[i].x);

                for (size_t i = 0; i < vPoints.size(); i++)
                    vToU += uMat.at<float>(vPoints[i].y, vPoints[i].x);

                uPointNum += uPoints.size();
                vPointNum += vPoints.size();
            }

            if (uPointNum == 0 || vPointNum == 0)
                return 1;
            else
                return (uToV / uPointNum + vToU / vPointNum) / 2.0;
        }

        inline vector<vector<Point>> OCM::GetChannels(const Mat& sketchImage, int orientNum)
        {
            int sigma = 9, lambda = 24, ksize = sigma * 6 + 1;
            vector<Mat> tmp(orientNum);

            for (int i = 0; i < orientNum; i++)
            {
                Mat kernel = getGaborKernel(Size(ksize, ksize), sigma, 
                    CV_PI / orientNum * i, lambda, 1, 0);

                filter2D(sketchImage, tmp[i], CV_64F, kernel);
                tmp[i] = abs(tmp[i]);
            }

            vector<Point> points = GetEdgels(sketchImage);

            vector<vector<Point>> channels(orientNum);
            for (int i = 0; i < points.size(); i++)
            {
                double maxResponse = -INF;
                int index = -1;

                for (int j = 0; j < orientNum; j++)
                {
                    if (tmp[j].at<double>(points[i].y, points[i].x) > maxResponse)
                    {
                        maxResponse = tmp[j].at<double>(points[i].y, points[i].x);
                        index = j;
                    }
                }

                assert(index >= 0 && index < orientNum);
                channels[index].push_back(points[i]);
            }

            return channels;
        }

        inline OCM::Info OCM::Transform(const Mat& sketchImage, double maxDistance)
        {
            vector<vector<Point>> channels = GetChannels(sketchImage, 6);
            Info result(channels.size());

            for (size_t i = 0; i < channels.size(); i++)
            {
                Mat dt(sketchImage.size(), CV_32F);

                for (int j = 0; j < dt.rows; j++)
                    for (int k = 0; k < dt.cols; k++)
                        dt.at<float>(j, k) = maxDistance * maxDistance;

                for (size_t j = 0; j < channels[i].size(); j++)
                {
                    int left = (int)floor(channels[i][j].x - maxDistance),
                        right = (int)ceil(channels[i][j].x + maxDistance),
                        top = (int)floor(channels[i][j].y - maxDistance),
                        bottom = (int)ceil(channels[i][j].y + maxDistance);
                    left = left < 0 ? 0 : left;
                    right = right > dt.cols ? dt.cols : right;
                    top = top < 0 ? 0 : top;
                    bottom = bottom > dt.rows ? dt.rows : bottom;

                    for (int m = top; m < bottom; m++)
                    {
                        for (int n = left; n < right; n++)
                        {
                            double distance = (m - channels[i][j].y) * (m - channels[i][j].y) + 
                                (n - channels[i][j].x) * (n - channels[i][j].x);

                            dt.at<float>(m, n) = min(distance, (double)dt.at<float>(m, n));
                        }
                    }
                }

                result[i] = CreateTuple(channels[i], dt);
            }

            return result;
        }

        ///////////////////////////////////////////////////////////////////////

        // Hitmap Matching
        class Hitmap : public Feature
        {
        public:
            typedef Vector<Tuple<vector<Point>, Mat>> Info;

            Info GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false);
            Info GetFeatureWithoutPreprocess(const Mat& sketchImage);

            static double GetDistance(const Info& u, const Info& v);
            static vector<vector<Point>> GetChannels(const Mat& sketchImage, int orientNum);

            virtual String GetName() const { return "hit"; };

        protected:
            virtual Info Transform(const Mat& sketchImage, double maxDistance = 22);
        };

        inline Hitmap::Info Hitmap::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning)
        {
            return Transform(Preprocess(sketchImage, thinning));
        }

        inline Hitmap::Info Hitmap::GetFeatureWithoutPreprocess(const Mat& sketchImage)
        {
            return Transform(sketchImage);
        }

        inline double Hitmap::GetDistance(const Info& u, const Info& v)
        {
            assert(u.size() == v.size());
            int orientNum = u.size(), uPointNum = 0, vPointNum = 0;
            double uToV = 0, vToU = 0;

            for (int i = 0; i < orientNum; i++)
            {
                const vector<Point>& uPoints = u[i].Item1();
                const vector<Point>& vPoints = v[i].Item1();
                const Mat& uMat = u[i].Item2();
                const Mat& vMat = v[i].Item2();

                for (size_t i = 0; i < uPoints.size(); i++)
                    uToV += vMat.at<uchar>(uPoints[i].y, uPoints[i].x);

                for (size_t i = 0; i < vPoints.size(); i++)
                    vToU += uMat.at<uchar>(vPoints[i].y, vPoints[i].x);

                uPointNum += uPoints.size();
                vPointNum += vPoints.size();
            }

            if (uPointNum == 0 || vPointNum == 0)
                return 1;
            else
                return 1 - sqrt((uToV / uPointNum) * (vToU / vPointNum));
        }

        inline vector<vector<Point>> Hitmap::GetChannels(const Mat& sketchImage, int orientNum)
        {
            int sigma = 9, lambda = 24, ksize = sigma * 6 + 1;
            vector<Mat> tmp(orientNum);

            for (int i = 0; i < orientNum; i++)
            {
                Mat kernel = getGaborKernel(Size(ksize, ksize), sigma, 
                    CV_PI / orientNum * i, lambda, 1, 0);

                filter2D(sketchImage, tmp[i], CV_64F, kernel);
                tmp[i] = abs(tmp[i]);
            }

            vector<Point> points = GetEdgels(sketchImage);

            vector<vector<Point>> channels(orientNum);
            for (int i = 0; i < points.size(); i++)
            {
                double maxResponse = -INF;
                int index = -1;

                for (int j = 0; j < orientNum; j++)
                {
                    if (tmp[j].at<double>(points[i].y, points[i].x) > maxResponse)
                    {
                        maxResponse = tmp[j].at<double>(points[i].y, points[i].x);
                        index = j;
                    }
                }

                assert(index >= 0 && index < orientNum);
                channels[index].push_back(points[i]);
            }

            return channels;
        }

        inline Hitmap::Info Hitmap::Transform(const Mat& sketchImage, double maxDistance)
        {
            vector<vector<Point>> channels = GetChannels(sketchImage, 6);
            Info result(channels.size());

            for (size_t i = 0; i < channels.size(); i++)
            {
                Mat dt(sketchImage.size(), CV_8U);
                dt = Scalar::all(0);

                for (size_t j = 0; j < channels[i].size(); j++)
                {
                    int left = (int)floor(channels[i][j].x - maxDistance),
                        right = (int)ceil(channels[i][j].x + maxDistance),
                        top = (int)floor(channels[i][j].y - maxDistance),
                        bottom = (int)ceil(channels[i][j].y + maxDistance);
                    left = left < 0 ? 0 : left;
                    right = right > dt.cols ? dt.cols : right;
                    top = top < 0 ? 0 : top;
                    bottom = bottom > dt.rows ? dt.rows : bottom;

                    for (int m = top; m < bottom; m++)
                    {
                        for (int n = left; n < right; n++)
                        {
                            double distance = sqrt((m - channels[i][j].y) * (m - channels[i][j].y) + 
                                (n - channels[i][j].x) * (n - channels[i][j].x));

                            if (distance <= maxDistance)
                                dt.at<uchar>(m, n) = 1;
                        }
                    }
                }

                result[i] = CreateTuple(channels[i], dt);
            }

            return result;
        }
    }
}
}