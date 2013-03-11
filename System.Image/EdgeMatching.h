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
        class ChamferMatching : public Feature
        {
        public:
            typedef tuple<vector<Point>, Mat> Info;

            Info GetFeatureWithPreprocess(const Mat& sketchImage, bool thinning = false) const;
            Info GetFeatureWithoutPreprocess(const Mat& sketchImage) const;

            static double GetDistance(const ChamferMatching::Info& u, const ChamferMatching::Info& v);

            virtual String GetName() const { return "cm"; };

        protected:
            virtual tuple<vector<Point>, Mat> Transform(const Mat& sketchImage,
                int maxDistance = 40) const;
        };

        inline ChamferMatching::Info ChamferMatching::GetFeatureWithPreprocess(
            const Mat& sketchImage, bool thinning) const
        {
            return Transform(Preprocess(sketchImage, thinning));
        }

        inline ChamferMatching::Info ChamferMatching::GetFeatureWithoutPreprocess(
            const Mat& sketchImage) const
        {
            return Transform(sketchImage);
        }

        inline double ChamferMatching::GetDistance(const Info& u, const Info& v)
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

        inline ChamferMatching::Info ChamferMatching::Transform(const Mat& sketchImage,
            int maxDistance) const
        {
            Mat result(sketchImage.size(), CV_64F);

            for (int i = 0; i < result.rows; i++)
                for (int j = 0; j < result.cols; j++)
                    result.at<double>(i, j) = maxDistance;

            vector<Point> points = GetEdgels(sketchImage);

            for (size_t i = 0; i < points.size(); i++)
            {
                for (int m = 0; m < result.rows; m++)
                {
                    for (int n = 0; n < result.cols; n++)
                    {
                        double distance = sqrt((m - points[i].y) * (m - points[i].y) + 
                            (n - points[i].x) * (n - points[i].x));
                        result.at<double>(m, n) = min(distance, result.at<double>(m, n));
                    }
                }
            }

            return make_tuple(points, result);
        }
    }
}