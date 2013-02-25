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
		// Assume: Edgels are 1 and Background is 0.
		inline tuple<vector<Point>, vector<Point>> FindJunctionsOrEndpoints(const Mat& binaryImage)
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
						for (int m = -1; m <= 1; m++)
						{
							for (int n = -1; n <= 1; n++)
							{
								int r = m + i, c = n + j;

								if (r < 0 || r >= binaryImage.rows || c < 0 || c >= binaryImage.cols)
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
							junctions.push_back(Point(j, i));
						else if (distance == 2)
							endPoints.push_back(Point(j, i));
					}
				}
			}

			return make_tuple(junctions, endPoints);
		}

        // Assume: Edgels are 1 and Background is 0.
        // The values are ordered from the top-left point going anti-clockwise around the pixel.
        inline vector<uchar> GetNeighbourValues(const Mat& binaryImage, int centreY, int centreX)
        {
            static int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
            static int dx[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
            vector<uchar> result;

            for (int i = 0; i < 8; i++)
            {
                int newY = centreY + dy[i], newX = centreX + dx[i];

                if (newY < 0 || newY >= binaryImage.rows || newX < 0 || newX >= binaryImage.cols)
                    result.push_back(0);
                else
                    result.push_back(binaryImage.at<uchar>(newY, newX));
            }

            return result;
        }

        // Assume: Edgels are 1 and Background is 0.
        Point NextPoint(const Mat& binaryImage, int centreY, int centreX)
        {
            return Point();
        }
	}
}