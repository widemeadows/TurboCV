#pragma once

#include "../System/Math.h"
#include "Util.h"
#include <cv.h>
#include <unordered_set>
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        class Geometry
        {
        public:
            static double EulerDistance(const Point& u, const Point& v);

            static ArrayList<double> EulerDistance(const Point& u, const ArrayList<Point>& vec);

            static double Angle(const Point& start, const Point& end);

            static ArrayList<double> Angle(const Point& start, const ArrayList<Point>& ends);
        };

        inline double Geometry::EulerDistance(const Point& u, const Point& v)
        {
            return sqrt((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y));
        }

        inline ArrayList<double> Geometry::EulerDistance(const Point& u, const ArrayList<Point>& vec)
        {
            ArrayList<double> distances(vec.Count());

            for (int i = 0; i < vec.Count(); i++)
                distances[i] = Geometry::EulerDistance(u, vec[i]);

            return distances;
        }

        inline double Geometry::Angle(const Point& start, const Point& end)
        {
            double deltaY = end.y - start.y;
            double deltaX = end.x - start.x;
            
            if (deltaX == 0 && deltaY == 0)
                return INF;

            double angle = atan2(deltaY, deltaX) + CV_PI;
            if (angle < 0)
                angle = 0;
            if (angle > 2 * CV_PI)
                angle = 2 * CV_PI;

            return angle;
        }

        inline ArrayList<double> Geometry::Angle(const Point& start, const ArrayList<Point>& ends)
        {
            ArrayList<double> angles(ends.Count());

            for (int i = 0; i < ends.Count(); i++)
                angles[i] = Geometry::Angle(start, ends[i]);

            return angles;
        }

        inline ArrayList<Point> SampleOnGrid(
            size_t height, 
            size_t width, 
            size_t numPerDirection)
        {
            int heightStep = height / numPerDirection, 
                widthStep = width / numPerDirection;
            ArrayList<Point> points;

            for (int i = heightStep / 2; i < height; i += heightStep)
                for (int j = widthStep / 2; j < width; j += widthStep)
                    points.Add(Point(j, i));

            return points;
        }

        inline ArrayList<Point> SampleFromPoints(
            const ArrayList<Point>& points, 
            size_t samplingNum)
        {
            size_t pointNum = points.Count();
            assert(pointNum >= samplingNum);

            ArrayList<Tuple<double, Tuple<Point, Point>>> distances(pointNum * (pointNum - 1) / 2);
            std::unordered_set<Point, PointHash> pivots;

            int counter = 0;
            for (size_t i = 0; i < pointNum; i++)
            {
                for (size_t j = i + 1; j < pointNum; j++)
                {
                    double distance = Geometry::EulerDistance(points[i], points[j]);
                    distances[counter++] = CreateTuple(distance, 
                        CreateTuple(points[i], points[j]));
                }
                pivots.insert(points[i]);
            }
            sort(distances.begin(), distances.end());

            int ptr = 0;
            while (pivots.size() > samplingNum)
            {
                Tuple<Point, Point> pointPair = distances[ptr++].Item2();
                if (pivots.find(pointPair.Item1()) != pivots.end() &&
                    pivots.find(pointPair.Item2()) != pivots.end())
                    pivots.erase(pointPair.Item2());
            }

            ArrayList<Point> results;
            for (auto pivot : pivots)
                results.Add(pivot);

            return results;
        }

        inline ArrayList<Point> SampleOnShape(const Mat& sketchImage, size_t samplingNum)
        {
            return SampleFromPoints(GetEdgels(sketchImage), samplingNum);
        }
    }
}
}