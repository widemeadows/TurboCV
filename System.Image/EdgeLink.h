#pragma once

#include <cv.h>
#include <tuple>
#include <vector>
using namespace cv;
using namespace std;

namespace System
{
	namespace Image
	{
		typedef tuple<int, int> Point;

		inline vector<Point> FindJunctionsOrEndPoints(const Mat& binaryImage)
		{
		}

		inline vector<Point> FindJunctions(const Mat& binaryImage)
		{
			for (int i = 0; i < binaryImage.rows; i++)
			{
				for (int j = 0; j < binaryImage.cols; j++)
				{

				}
			}
		}
	}
}