#pragma once

#include <cv.h>
using namespace cv;

namespace System
{
    namespace Image
    {
        class Geometry
        {
        public:
            static double EulerDistance(const Point& u, const Point& v);

            static double Angle(const Point& start, const Point& end);
        };

        inline double Geometry::EulerDistance(const Point& u, const Point& v)
        {
            return sqrt((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y));
        }

        inline double Geometry::Angle(const Point& start, const Point& end)
        {
	        double deltaY = end.y - start.y;
	        double deltaX = end.x - start.x;
	        assert(deltaX != 0 || deltaY != 0);

	        double angle = atan2(deltaY, deltaX) + CV_PI;
	        if (angle < 0)
		        angle = 0;
	        if (angle > 2 * CV_PI)
		        angle = 2 * CV_PI;

	        return angle;
        }
    }
}