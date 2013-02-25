#pragma once

#include "Util.h"
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

		// Assume: Edgels are 1 and Background is 0.
		inline tuple<vector<Point>, vector<Point>> FindJunctionsOrEndPoints(const Mat& binaryImage)
		{
			vector<Point> junctions, endPoints;
			Mat patch(3, 3, CV_8U);
			vector<int> a(8), b(8);

			for (int i = 0; i < binaryImage.rows; i++)
			{
				for (int j = 0; j < binaryImage.cols; j++)
				{
					if (binaryImage.at<uchar>(i, j))
					{
						// Pixels in the 3x3 patch are numbered as follows:
						// 0 3 6
						// 1 4 7
						// 2 5 8
						for (int m = -1; m <= 1; m++)
						{
							for (int n = -1; n <= 1; n++)
							{
								int r = m + i, c = n + j;

								if (r < 0 || r > binaryImage.rows || c < 0 || c > binaryImage.cols)
									patch.at<uchar>(m + 1, n + 1) = 0;
								else
									patch.at<uchar>(m + 1, n + 1) = binaryImage.at<uchar>(r, c);
							}
						}

						a[0] = patch.at<uchar>(0, 0);
						a[1] = patch.at<uchar>(1, 0);
						a[2] = patch.at<uchar>(2, 0);
						a[3] = patch.at<uchar>(2, 1);
						a[4] = patch.at<uchar>(2, 2);
						a[5] = patch.at<uchar>(1, 2);
						a[6] = patch.at<uchar>(0, 2);
						a[7] = patch.at<uchar>(0, 1);
						for (int m = 0; m < 7; m++)
							b[m] = a[m + 1];
						b[7] = a[0];

						double distance = NormOneDistance(a, b);
						if (distance >= 6)
							junctions.push_back(make_tuple(i, j));
						else if (distance == 2)
							junctions.push_back(make_tuple(i, j));
					}
				}
			}

			return make_tuple(junctions, endPoints);
		}
	}
}