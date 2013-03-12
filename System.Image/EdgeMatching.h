#pragma once

#include "Feature.h"
#include <tuple>
#include <cv.h>
using namespace std;
using namespace cv;

namespace System
{
    namespace Image
    {
        // Chamfer Matching
        class CM : public Feature
        {
        public:
            typedef tuple<vector<Point>, Mat> Info;

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
            const vector<Point>& uPoints = get<0>(u);
            const vector<Point>& vPoints = get<0>(v);
            const Mat& uMat = get<1>(u);
            const Mat& vMat = get<1>(v);
            double uToV = 0, vToU = 0;

            for (size_t i = 0; i < uPoints.size(); i++)
                uToV += vMat.at<double>(uPoints[i].y, uPoints[i].x);

            for (size_t i = 0; i < vPoints.size(); i++)
                vToU += uMat.at<double>(vPoints[i].y, vPoints[i].x);

            return (uToV / uPoints.size() + vToU / vPoints.size()) / 2.0;
        }

        inline CM::Info CM::Transform(const Mat& sketchImage, double maxDistance) const
        {
            Mat dt(sketchImage.size(), CV_64F);

            for (int i = 0; i < dt.rows; i++)
                for (int j = 0; j < dt.cols; j++)
                    dt.at<double>(i, j) = maxDistance;

            vector<Point> points = GetEdgels(sketchImage);

            for (size_t i = 0; i < points.size(); i++)
            {
                for (int m = 0; m < dt.rows; m++)
                {
                    for (int n = 0; n < dt.cols; n++)
                    {
                        double distance = sqrt((m - points[i].y) * (m - points[i].y) + 
                            (n - points[i].x) * (n - points[i].x));
                        dt.at<double>(m, n) = min(distance, dt.at<double>(m, n));
                    }
                }
            }

            return make_tuple(points, dt);
        }

        ///////////////////////////////////////////////////////////////////////

        // Oriented Chamfer Matching
        class OCM : public Feature
        {
        public:
            typedef vector<tuple<vector<Point>, Mat>> Info;

            Info GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false) const;
            Info GetFeatureWithoutPreprocess(const Mat& sketchImage) const;

            static double GetDistance(const Info& u, const Info& v);

            static vector<vector<Point>> GetChannels(const Mat& sketchImage, int orientNum);

            virtual String GetName() const { return "ocm"; };

        protected:
            virtual Info Transform(const Mat& sketchImage, double maxDistance = 40) const;
        };

        inline OCM::Info OCM::GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning) const
        {
            return Transform(Preprocess(sketchImage, thinning));
        }

        inline OCM::Info OCM::GetFeatureWithoutPreprocess(const Mat& sketchImage) const
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
                const vector<Point>& uPoints = get<0>(u[i]);
                const vector<Point>& vPoints = get<0>(v[i]);
                const Mat& uMat = get<1>(u[i]);
                const Mat& vMat = get<1>(v[i]);

                for (size_t i = 0; i < uPoints.size(); i++)
                    uToV += vMat.at<float>(uPoints[i].y, uPoints[i].x);

                for (size_t i = 0; i < vPoints.size(); i++)
                    vToU += uMat.at<float>(vPoints[i].y, vPoints[i].x);

                uPointNum += uPoints.size();
                vPointNum += vPoints.size();
            }

            return (uToV / uPointNum + vToU / vPointNum) / 2.0;
        }

        inline vector<vector<Point>> OCM::GetChannels(const Mat& sketchImage, int orientNum)
        {
            int sigma = 9, lambda = 24, ksize = sigma * 6 + 1;
            vector<Mat> gaborResponses;

            for (int i = 0; i < orientNum; i++)
            {
                Mat kernel = getGaborKernel(Size(ksize, ksize), sigma, 
                    CV_PI / orientNum * i, lambda, 1, 0);

                Mat gaborResponse;
                filter2D(sketchImage, gaborResponse, CV_64F, kernel);

                gaborResponses.push_back(abs(gaborResponse));
            }

            vector<Point> points = GetEdgels(sketchImage);

            vector<vector<Point>> channels(orientNum);
            for (int i = 0; i < points.size(); i++)
            {
                double maxResponse = -INF;
                int index = -1;

                for (int j = 0; j < orientNum; j++)
                {
                    if (gaborResponses[j].at<double>(points[i].y, points[i].x) > maxResponse)
                    {
                        maxResponse = gaborResponses[j].at<double>(points[i].y, points[i].x);
                        index = j;
                    }
                }

                assert(index >= 0 && index < orientNum);
                channels[index].push_back(points[i]);
            }

            return channels;
        }

        inline OCM::Info OCM::Transform(const Mat& sketchImage, double maxDistance) const
        {
            Info result;
            vector<vector<Point>> channels = GetChannels(sketchImage, 6);

            for (int i = 0; i < channels.size(); i++)
            {
                Mat dt(sketchImage.size(), CV_32F);

                for (int j = 0; j < dt.rows; j++)
                    for (int k = 0; k < dt.cols; k++)
                        dt.at<float>(j, k) = maxDistance;

                for (size_t j = 0; j < channels[i].size(); j++)
                {
                    for (int m = 0; m < dt.rows; m++)
                    {
                        for (int n = 0; n < dt.cols; n++)
                        {
                            double distance = sqrt((m - channels[i][j].y) * (m - channels[i][j].y) + 
                                (n - channels[i][j].x) * (n - channels[i][j].x));
                            dt.at<float>(m, n) = min(distance, (double)dt.at<float>(m, n));
                        }
                    }
                }

                result.push_back(make_tuple(channels[i], dt));
            }

            return result;
        }
    }
}