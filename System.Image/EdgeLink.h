#pragma once

#include "Util.h"
#include <cv.h>
#include <tuple>
#include <vector>
#include <unordered_set>
using namespace cv;
using namespace std;

namespace System
{
	namespace Image
	{
		typedef vector<Point> Edge;

		struct PointHash
		{
			size_t operator()(const Point& v) const
			{
				return v.y * 10000000 + v.x;
			}
		};

		enum Status
		{
			NormalPoint,
			LastPoint,
			NoPoint,
		};

        // The values are ordered from the top-left point going anti-clockwise around the pixel.
		template<typename T>
        inline vector<T> GetNeighbourValues(const Mat& image, const Point& centre)
        {
            static int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
            static int dx[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
            vector<T> result;

            for (int i = 0; i < 8; i++)
            {
                int newY = centre.y + dy[i], newX = centre.x + dx[i];

                if (newY < 0 || newY >= image.rows || newX < 0 || newX >= image.cols)
                    result.push_back(0);
                else
                    result.push_back(image.at<T>(newY, newX));
            }

            return result;
        }

		// Assume: Edgels are 1 and Background is 0.
		inline tuple<vector<Point>, vector<Point>> FindJunctionsOrEndpoints(const Mat& binaryImage)
		{
			vector<Point> junctions, endPoints;
			Mat patch(3, 3, CV_8U);
			vector<uchar> a(8), b(8);

			for (int i = 0; i < binaryImage.rows; i++)
			{
				for (int j = 0; j < binaryImage.cols; j++)
				{
					if (binaryImage.at<uchar>(i, j))
					{
						a = GetNeighbourValues<uchar>(binaryImage, Point(j, i));
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

		// Find next connected edge point.
        // Assume: Edgels are 1 and Background is 0.
        tuple<Status, Point> NextPoint(const Mat& edgeFlags, int edgeNo, 
			const unordered_set<Point, PointHash>& junctions,
			const unordered_set<Point, PointHash>& endpoints, 
			const Point& centre)
        {
            static int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
            static int dx[] = { -1, -1, -1, 0, 1, 1, 1, 0 };

			for (int i = 0; i < 8; i++)
			{
				int newY = centre.y + dy[i], newX = centre.x + dx[i];

				if (newY >= 0 && newY < edgeFlags.rows && newX >= 0 && newX < edgeFlags.cols)
				{
					// Chech whether it is a junction that hasn't been marked 
					// as part of the current edge.
					if (junctions.find(Point(newX, newY)) != junctions.end() &&
						edgeFlags.at<int>(newY, newX) != -edgeNo)
						return make_tuple(LastPoint, Point(newX, newY));
				}
			}

			// If we get here there were no junction points. Search through neighbours
			// and return first connected edge point that itself has less than two
			// neighbours connected back to our current edge. This prevents occasional
			// erroneous doubling back onto the wrong segment.
			bool flag = false;
			Point record;

			for (int i = 0; i < 8; i++)
			{
				int newY = centre.y + dy[i], newX = centre.x + dx[i];

				if (newY >= 0 && newY < edgeFlags.rows && newX >= 0 && newX < edgeFlags.cols)
				{
					if (edgeFlags.at<int>(newY, newX) == 1)
					{
						vector<int> values = GetNeighbourValues<int>(edgeFlags, Point(newX, newY));
						int counter = 0;

						for (int j = values.size() - 1; j >= 0; j--)
							if (values[j] == -edgeNo)
								counter++;

						if (counter < 2)
							return make_tuple(NormalPoint, Point(newX, newY));
						else
						{
							flag = true;
							record.y = newY;
							record.x = newX;
						}
					}
				}
			}

			// If we get here (and 'flag' is true) there was no connected edge point
			// that had less than two connections to our current edge, but there was one
			// with more. Use the point we remembered above.
			if (flag)
				return make_tuple(NormalPoint, record);
			
			// If we get here there was no connecting next point at all.
			return make_tuple(NoPoint, record);
        }

		Edge TrackEdge(Mat& edgeFlags, int edgeNo, 
			const unordered_set<Point, PointHash>& junctions,
			const unordered_set<Point, PointHash>& endpoints,
			const Point& start)
		{
			Edge edge;
			edge.push_back(start);
			edgeFlags.at<int>(start.y, start.x) = -edgeNo;

			tuple<Status, Point> next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, start);
			Status status = get<0>(next);
			Point nextPoint = get<1>(next);

			while (status != NoPoint)
			{
				edge.push_back(nextPoint);
				edgeFlags.at<int>(nextPoint.y, nextPoint.x) = -edgeNo;

				if (status == LastPoint) // hit a junction
					status = NoPoint;
				else
				{
					next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, nextPoint);
					status = get<0>(next);
					nextPoint = get<1>(next);
				}
			}

			// Now track from original point in the opposite direction - but only if
			// the starting point was not a junction.
			if (junctions.find(start) == junctions.end())
			{
				reverse(edge.begin(), edge.end());

				next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, start);
				status = get<0>(next);
				nextPoint = get<1>(next);

				while (status != NoPoint)
				{
					edge.push_back(nextPoint);
					edgeFlags.at<int>(nextPoint.y, nextPoint.x) = -edgeNo;

					if (status == LastPoint)
						status = NoPoint;
					else
					{
						next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, nextPoint);
						status = get<0>(next);
						nextPoint = get<1>(next);
					}
				}
			}

			// Final check to see if this edge should have start and end points matched 
			// to form a loop. If the number of points in the list is four or more (the 
			// minimum number that could form a loop), and the endpoints are within a pixel 
			// of each other, append a copy if the first point to the end to complete the loop.
			if (edge.size() >= 4)
			{
				Point start = edge.front();
				Point end = edge.back();

				if (abs(start.y - end.y) <= 1 && abs(start.x - end.x) <= 2)
					edge.push_back(start);
			}

			return edge;
		}

		// Assume: Edgels are 1 and Background is 0. 
		// We strongly recommend removing any isolated pixel before applying EdgeLink.
		vector<Edge> EdgeLink(const Mat& binaryImage, int minLength = 10)
		{
			tuple<vector<Point>, vector<Point>> juncOrEnd = 
				FindJunctionsOrEndpoints(binaryImage);

			unordered_set<Point, PointHash> junctions, endpoints;
			for (auto p : get<0>(juncOrEnd))
				junctions.insert(p);
			for (auto p : get<1>(juncOrEnd))
				endpoints.insert(p);

			Mat edgeFlags;
			binaryImage.convertTo(edgeFlags, CV_32S);

			vector<Edge> edgeList;

			for (int i = 0; i < edgeFlags.rows; i++)
			{
				for (int j = 0; j < edgeFlags.cols; j++)
				{
					if (edgeFlags.at<int>(i, j) == 1)
					{
						Edge edge = TrackEdge(edgeFlags, edgeList.size() + 1, junctions, 
							endpoints, Point(j, i));

						if (edge.size() >= minLength)
							edgeList.push_back(edge);
					}
				}
			}

			return edgeList;
		}
	}
}