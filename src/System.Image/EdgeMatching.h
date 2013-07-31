#pragma once

#include "../System/System.h"
#include "Core.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        // Chamfer Matching
        //class CM : public Feature
        //{
        //public:
        //    typedef Group<ArrayList<Point>, Mat> Info;

        //    Info GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false,
        //        Size size = Size(256, 256));
        //    Info GetFeatureWithoutPreprocess(const Mat& sketchImage);

        //    static double GetDistance(const Info& u, const Info& v);

        //    virtual String GetName() const { return "cm"; };

        //protected:
        //    virtual Info Transform(const Mat& sketchImage, double maxDistance) const;
        //};

        //inline CM::Info CM::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning,
        //    Size size)
        //{
        //    double maxDistance = 40 * size.height / 256;
        //    return Transform(Preprocess(sketchImage, thinning, size), maxDistance);
        //}

        //inline CM::Info CM::GetFeatureWithoutPreprocess(const Mat& sketchImage)
        //{
        //    double maxDistance = 40 * sketchImage.rows / 256;
        //    return Transform(sketchImage, maxDistance);
        //}

        //inline double CM::GetDistance(const Info& u, const Info& v)
        //{
        //    const ArrayList<Point>& uPoints = u.Item1();
        //    const ArrayList<Point>& vPoints = v.Item1();
        //    const Mat& uMat = u.Item2();
        //    const Mat& vMat = v.Item2();
        //    double uToV = 0, vToU = 0;

        //    for (size_t i = 0; i < uPoints.Count(); i++)
        //        uToV += vMat.at<double>(uPoints[i].y, uPoints[i].x);

        //    for (size_t i = 0; i < vPoints.Count(); i++)
        //        vToU += uMat.at<double>(vPoints[i].y, vPoints[i].x);

        //    if (uPoints.Count() == 0 || vPoints.Count() == 0)
        //        return numeric_limits<double>::max();
        //    else
        //        return (uToV / uPoints.Count() + vToU / vPoints.Count()) / 2.0;
        //}

        //inline CM::Info CM::Transform(const Mat& sketchImage, double maxDistance) const
        //{
        //    Mat dt(sketchImage.size(), CV_64F);

        //    for (int i = 0; i < dt.rows; i++)
        //        for (int j = 0; j < dt.cols; j++)
        //            dt.at<double>(i, j) = maxDistance * maxDistance;

        //    ArrayList<Point> points = GetEdgels(sketchImage);

        //    for (size_t i = 0; i < points.Count(); i++)
        //    {
        //        int left = (int)floor(points[i].x - maxDistance),
        //            right = (int)ceil(points[i].x + maxDistance),
        //            top = (int)floor(points[i].y - maxDistance),
        //            bottom = (int)ceil(points[i].y + maxDistance);
        //        left = left < 0 ? 0 : left;
        //        right = right > dt.cols ? dt.cols : right;
        //        top = top < 0 ? 0 : top;
        //        bottom = bottom > dt.rows ? dt.rows : bottom;

        //        for (int m = top; m < bottom; m++)
        //        {
        //            for (int n = left; n < right; n++)
        //            {
        //                double distance = (m - points[i].y) * (m - points[i].y) + 
        //                    (n - points[i].x) * (n - points[i].x);

        //                dt.at<double>(m, n) = min(distance, dt.at<double>(m, n));
        //            }
        //        }
        //    }

        //    return CreateGroup(points, dt);
        //}

        /////////////////////////////////////////////////////////////////////////

        //// Oriented Chamfer Matching
        //class OCM : public Feature
        //{
        //public:
        //    typedef ArrayList<Group<ArrayList<Point>, Mat>> Info;

        //    Info GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false,
        //        Size size = Size(256, 256));
        //    Info GetFeatureWithoutPreprocess(const Mat& sketchImage);

        //    static double GetDistance(const Info& u, const Info& v);
        //    static ArrayList<ArrayList<Point>> GetChannels(const Mat& sketchImage, int orientNum);

        //    virtual String GetName() const { return "ocm"; };

        //protected:
        //    virtual Info Transform(const Mat& sketchImage, double maxDistance);
        //};

        //inline OCM::Info OCM::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning,
        //    Size size)
        //{
        //    double maxDistance = 40 * size.height / 256;
        //    return Transform(Preprocess(sketchImage, thinning), maxDistance);
        //}

        //inline OCM::Info OCM::GetFeatureWithoutPreprocess(const Mat& sketchImage)
        //{
        //    double maxDistance = 40 * sketchImage.rows / 256;
        //    return Transform(sketchImage, maxDistance);
        //}

        //inline double OCM::GetDistance(const Info& u, const Info& v)
        //{
        //    assert(u.Count() == v.Count());
        //    int orientNum = u.Count(), uPointNum = 0, vPointNum = 0;
        //    double uToV = 0, vToU = 0;

        //    for (int i = 0; i < orientNum; i++)
        //    {
        //        const ArrayList<Point>& uPoints = u[i].Item1();
        //        const ArrayList<Point>& vPoints = v[i].Item1();
        //        const Mat& uMat = u[i].Item2();
        //        const Mat& vMat = v[i].Item2();

        //        for (size_t i = 0; i < uPoints.Count(); i++)
        //            uToV += vMat.at<float>(uPoints[i].y, uPoints[i].x);

        //        for (size_t i = 0; i < vPoints.Count(); i++)
        //            vToU += uMat.at<float>(vPoints[i].y, vPoints[i].x);

        //        uPointNum += uPoints.Count();
        //        vPointNum += vPoints.Count();
        //    }

        //    if (uPointNum == 0 || vPointNum == 0)
        //        return numeric_limits<double>::max();
        //    else
        //        return (uToV / uPointNum + vToU / vPointNum) / 2.0;
        //}

        //inline ArrayList<ArrayList<Point>> OCM::GetChannels(const Mat& sketchImage, int orientNum)
        //{
        //    int sigma = 9, lambda = 24, ksize = sigma * 6 + 1;
        //    ArrayList<Mat> tmp(orientNum);

        //    for (int i = 0; i < orientNum; i++)
        //    {
        //        Mat kernel = getGaborKernel(Size(ksize, ksize), sigma, 
        //            CV_PI / orientNum * i, lambda, 1, 0);

        //        filter2D(sketchImage, tmp[i], CV_64F, kernel);
        //        tmp[i] = abs(tmp[i]);
        //    }

        //    ArrayList<Point> points = GetEdgels(sketchImage);

        //    ArrayList<ArrayList<Point>> channels(orientNum);
        //    for (int i = 0; i < points.Count(); i++)
        //    {
        //        double maxResponse = -INF;
        //        int index = -1;

        //        for (int j = 0; j < orientNum; j++)
        //        {
        //            if (tmp[j].at<double>(points[i].y, points[i].x) > maxResponse)
        //            {
        //                maxResponse = tmp[j].at<double>(points[i].y, points[i].x);
        //                index = j;
        //            }
        //        }

        //        assert(index >= 0 && index < orientNum);
        //        channels[index].Add(points[i]);
        //    }

        //    return channels;
        //}

        //inline OCM::Info OCM::Transform(const Mat& sketchImage, double maxDistance)
        //{
        //    ArrayList<ArrayList<Point>> channels = GetChannels(sketchImage, 6);
        //    Info result(channels.Count());

        //    for (size_t i = 0; i < channels.Count(); i++)
        //    {
        //        Mat dt(sketchImage.size(), CV_32F);

        //        for (int j = 0; j < dt.rows; j++)
        //            for (int k = 0; k < dt.cols; k++)
        //                dt.at<float>(j, k) = maxDistance * maxDistance;

        //        for (size_t j = 0; j < channels[i].Count(); j++)
        //        {
        //            int left = (int)floor(channels[i][j].x - maxDistance),
        //                right = (int)ceil(channels[i][j].x + maxDistance),
        //                top = (int)floor(channels[i][j].y - maxDistance),
        //                bottom = (int)ceil(channels[i][j].y + maxDistance);
        //            left = left < 0 ? 0 : left;
        //            right = right > dt.cols ? dt.cols : right;
        //            top = top < 0 ? 0 : top;
        //            bottom = bottom > dt.rows ? dt.rows : bottom;

        //            for (int m = top; m < bottom; m++)
        //            {
        //                for (int n = left; n < right; n++)
        //                {
        //                    double distance = (m - channels[i][j].y) * (m - channels[i][j].y) + 
        //                        (n - channels[i][j].x) * (n - channels[i][j].x);

        //                    dt.at<float>(m, n) = min(distance, (double)dt.at<float>(m, n));
        //                }
        //            }
        //        }

        //        result[i] = CreateGroup(channels[i], dt);
        //    }

        //    return result;
        //}

        ///////////////////////////////////////////////////////////////////////

        // Hitmap Matching
        class Hitmap
        {
        public:
            typedef ArrayList<Group<ArrayList<cv::Point>, cv::Mat>> Info;

            Info GetFeature(const cv::Mat& sketchImage);

            static double GetDistance(const Info& u, const Info& v);
            static ArrayList<ArrayList<cv::Point>> GetChannels(const cv::Mat& sketchImage, int orientNum);

            virtual TString GetName() const { return "hit"; };

        protected:
            virtual Info Transform(const cv::Mat& sketchImage, double maxDistance);
        };

        inline Hitmap::Info Hitmap::GetFeature(const cv::Mat& sketchImage)
        {
            double maxDistance = 22;
            return Transform(sketchImage, maxDistance);
        }

        inline double Hitmap::GetDistance(const Info& u, const Info& v)
        {
            assert(u.Count() == v.Count());
            int orientNum = u.Count(), uPointNum = 0, vPointNum = 0;
            double uToV = 0, vToU = 0;

            for (int i = 0; i < orientNum; i++)
            {
                const ArrayList<cv::Point>& uPoints = u[i].Item1();
                const ArrayList<cv::Point>& vPoints = v[i].Item1();
                const cv::Mat& uMat = u[i].Item2();
                const cv::Mat& vMat = v[i].Item2();

                for (size_t i = 0; i < uPoints.Count(); i++)
                    uToV += vMat.at<uchar>(uPoints[i].y, uPoints[i].x);

                for (size_t i = 0; i < vPoints.Count(); i++)
                    vToU += uMat.at<uchar>(vPoints[i].y, vPoints[i].x);

                uPointNum += uPoints.Count();
                vPointNum += vPoints.Count();
            }

            if (uPointNum == 0 || vPointNum == 0)
                return 1;
            else
                return 1 - sqrt((uToV / uPointNum) * (vToU / vPointNum));
        }

        inline ArrayList<ArrayList<cv::Point>> Hitmap::GetChannels(const cv::Mat& sketchImage, int orientNum)
        {
            int sigma = 9, lambda = 24, ksize = sigma * 6 + 1;
            ArrayList<cv::Mat> tmp(orientNum);

            for (int i = 0; i < orientNum; i++)
            {
                cv::Mat kernel = getGaborKernel(cv::Size(ksize, ksize), sigma, 
                    CV_PI / orientNum * i, lambda, 1, 0);

                filter2D(sketchImage, tmp[i], CV_64F, kernel);
                tmp[i] = abs(tmp[i]);
            }

            ArrayList<cv::Point> points = GetEdgels(sketchImage);

            ArrayList<ArrayList<cv::Point>> channels(orientNum);
            for (int i = 0; i < points.Count(); i++)
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
                channels[index].Add(points[i]);
            }

            return channels;
        }

        inline Hitmap::Info Hitmap::Transform(const cv::Mat& sketchImage, double maxDistance)
        {
            ArrayList<ArrayList<cv::Point>> channels = GetChannels(sketchImage, 6);
            Info result(channels.Count());

            for (size_t i = 0; i < channels.Count(); i++)
            {
                cv::Mat dt(sketchImage.size(), CV_8U);
                dt = cv::Scalar::all(0);

                for (size_t j = 0; j < channels[i].Count(); j++)
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

                result[i] = CreateGroup(channels[i], dt);
            }

            return result;
        }
    }
}
}