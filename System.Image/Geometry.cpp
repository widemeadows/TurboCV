#include "../System/System.h"
#include "System.Image.h"
#include <cv.h>
using namespace cv;

namespace TurboCV
{
namespace System
{
	namespace Image
	{
		double EulerDistance(const Point& u, const Point& v)
		{
			return sqrt((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y));
		}

		ArrayList<double> EulerDistance(const Point& u, const ArrayList<Point>& vec)
		{
			ArrayList<double> distances(vec.Count());

			for (int i = 0; i < vec.Count(); i++)
				distances[i] = EulerDistance(u, vec[i]);

			return distances;
		}

		double Angle(const Point& start, const Point& end)
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

		ArrayList<double> Angle(const Point& start, const ArrayList<Point>& ends)
		{
			ArrayList<double> angles(ends.Count());

			for (int i = 0; i < ends.Count(); i++)
				angles[i] = Angle(start, ends[i]);

			return angles;
		}
	}
}
}