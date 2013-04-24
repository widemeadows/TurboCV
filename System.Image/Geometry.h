#pragma once

#include "../System/Math.h"
#include "Util.h"
#include <cv.h>
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
            ArrayList<double> distances(vec.size());

            for (int i = 0; i < vec.size(); i++)
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
            ArrayList<double> angles(ends.size());

            for (int i = 0; i < ends.size(); i++)
                angles[i] = Geometry::Angle(start, ends[i]);

            return angles;
        }
    }
}
}