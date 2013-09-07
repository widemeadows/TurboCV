#include "../System/System.h"
#include "Core.h"
#include <cv.h>
#include <queue>
#include <unordered_set>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
	namespace Image
	{
		//////////////////////////////////////////////////////////////////////////
		// APIs for Morphological Operations
		//////////////////////////////////////////////////////////////////////////

		void thin(InputArray input, OutputArray output, int iterations) 
		{
			assert(input.type() == CV_8U);

			Mat src = input.getMat();
			src.copyTo(output);

			Mat dst = output.getMat();
			for (int n = 0; n < iterations; n++) 
			{
				Mat prev = dst.clone();
				for (int i = 0; i < src.rows; i++) 
				{
					for (int j = 0; j < src.cols; j++) 
					{
						if (prev.at<uchar>(i, j)) 
						{
							int ap = 0;
							int p2 = (i == 0) ? 0 : prev.at<uchar>(i - 1, j) > 0;
							int p3 = (i == 0 || j == src.cols - 1) ? 0 : prev.at<uchar>(i - 1, j + 1) > 0;
							if (p2 == 0 && p3 == 1) 
								ap++;

							int p4 = (j == src.cols - 1) ? 0 : prev.at<uchar>(i, j + 1) > 0;
							if (p3 == 0 && p4 == 1) 
								ap++;

							int p5 = (i == src.rows - 1 || j == src.cols - 1) ? 0 : prev.at<uchar>(i + 1, j + 1) > 0;
							if (p4 == 0 && p5 == 1) 
								ap++;

							int p6 = (i == src.rows - 1) ? 0 : prev.at<uchar>(i + 1, j) > 0;
							if (p5 == 0 && p6 == 1) 
								ap++;

							int p7 = (i == src.rows - 1 || j == 0) ? 0 : prev.at<uchar>(i + 1, j - 1) > 0;
							if (p6 == 0 && p7 == 1) 
								ap++;

							int p8 = (j == 0) ? 0 : prev.at<uchar>(i, j - 1) > 0;
							if (p7 == 0 && p8 == 1) 
								ap++;

							int p9 = (i == 0 || j == 0) ? 0 : prev.at<uchar>(i - 1, j - 1) > 0;
							if (p8 == 0 && p9 == 1) 
								ap++;
							if (p9 == 0 && p2 == 1) 
								ap++;

							if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7)
								if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
									dst.at<uchar>(i, j) = 0;
						}
					}
				}

				prev = dst.clone();
				for (int i = 0; i < src.rows; i++) 
				{
					for (int j = 0; j < src.cols; j++)
					{
						if (prev.at<uchar>(i, j))
						{
							int ap = 0;
							int p2 = (i == 0) ? 0 : prev.at<uchar>(i - 1, j) > 0;
							int p3 = (i == 0 || j == src.cols - 1) ? 0 : prev.at<uchar>(i - 1, j + 1) > 0;
							if (p2 == 0 && p3 == 1) 
								ap++;

							int p4 = (j == src.cols - 1) ? 0 : prev.at<uchar>(i, j + 1) > 0;
							if (p3 == 0 && p4 == 1) 
								ap++;

							int p5 = (i == src.rows - 1 || j == src.cols - 1) ? 0 : prev.at<uchar>(i + 1, j + 1) > 0;
							if (p4 == 0 && p5 == 1) 
								ap++;

							int p6 = (i == src.rows - 1) ? 0 : prev.at<uchar>(i + 1, j) > 0;
							if (p5 == 0 && p6 == 1) 
								ap++;

							int p7 = (i == src.rows - 1 || j == 0) ? 0 : prev.at<uchar>(i + 1, j - 1) > 0;
							if (p6 == 0 && p7 == 1) 
								ap++;

							int p8 = (j == 0) ? 0 : prev.at<uchar>(i, j - 1) > 0;
							if (p7 == 0 && p8 == 1) 
								ap++;

							int p9 = (i == 0 || j == 0) ? 0 : prev.at<uchar>(i - 1, j - 1) > 0;
							if (p8 == 0 && p9 == 1) 
								ap++;
							if (p9 == 0 && p2 == 1) 
								ap++;

							if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7) 
								if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) 
									dst.at<uchar>(i, j) = 0;
						}
					}
				}
			}
		}

		void clean(InputArray input, OutputArray output, int nPoint)
		{
			assert(input.type() == CV_8U);

			Mat src = input.getMat();
			src.copyTo(output);

			Mat dst = output.getMat();
			Mat cur = src.clone();

			static int dy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
			static int dx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

			for (int i = 0; i < cur.rows; i++)
			{
				for (int j = 0; j < cur.cols; j++)
				{
					if (cur.at<uchar>(i, j))
					{
						queue<Point> q;
						vector<Point> component;

						q.push(Point(j, i));
						cur.at<uchar>(i, j) = 0;

						while (!q.empty())
						{
							Point front = q.front();
							q.pop();
							component.push_back(front);

							for (int k = 0; k < 8; k++)
							{
								int newY = front.y + dy[k], newX = front.x + dx[k];

								if (newY >= 0 && newY < cur.rows && 
									newX >= 0 && newX < cur.cols &&
									cur.at<uchar>(newY, newX))
								{
									q.push(Point(newX, newY));
									cur.at<uchar>(newY, newX) = 0;
								}
							}
						}

						if (component.size() <= nPoint)
						{
							for (size_t k = 0; k < component.size(); k++)
								dst.at<uchar>(component[k].y, component[k].x) = 0;
						}
					}
				}
			}
		}


		//////////////////////////////////////////////////////////////////////////
		// Edge Link
		//////////////////////////////////////////////////////////////////////////

		enum LinkStatus
		{
			NormalPoint,
			LastPoint,
			NoPoint,
		};

		// The values are ordered from the top-left point going anti-clockwise around the pixel.
		template<typename T>
		inline ArrayList<T> GetNeighbourValues(const Mat& image, const Point& centre)
		{
			static int dy[] = { -1, 0, 1, 1, 1, 0, -1, -1 };
			static int dx[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
			ArrayList<T> result;

			for (int i = 0; i < 8; i++)
			{
				int newY = centre.y + dy[i], newX = centre.x + dx[i];

				if (newY < 0 || newY >= image.rows || newX < 0 || newX >= image.cols)
					result.Add(0);
				else
					result.Add(image.at<T>(newY, newX));
			}

			return result;
		}

		inline Group<ArrayList<Point>, ArrayList<Point>> FindJunctionsOrEndpoints(
			const Mat& binaryImage)
		{
			ArrayList<Point> junctions, endPoints;
			Mat patch(3, 3, CV_8U);
			ArrayList<uchar> a(8), b(8);

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

						double distance = Math::NormOneDistance(a, b);
						if (distance >= 6)
							junctions.Add(Point(j, i));
						else if (distance == 2)
							endPoints.Add(Point(j, i));
					}
				}
			}

			return CreateGroup(junctions, endPoints);
		}

		// Find next connected edge point.
		Group<LinkStatus, Point> NextPoint(const Mat& edgeFlags, int edgeNo, 
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
					// Chech whether it is a junction/endpoint that hasn't been marked 
					// as part of the current edge.
					if ((junctions.find(Point(newX, newY)) != junctions.end() ||
						endpoints.find(Point(newX, newY)) != endpoints.end()) &&
						edgeFlags.at<int>(newY, newX) != -edgeNo)
						return CreateGroup(LastPoint, Point(newX, newY));
				}
			}

			// If we get here there were no junctions/endpoints. Search through neighbours
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
					if (edgeFlags.at<int>(newY, newX))
					{
						ArrayList<int> values = 
							GetNeighbourValues<int>(edgeFlags, Point(newX, newY));
						int counter = 0;

						for (int j = values.Count() - 1; j >= 0; j--)
							if (values[j] == -edgeNo)
								counter++;

						if (counter < 2)
							return CreateGroup(NormalPoint, Point(newX, newY));
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
			// that had less than two connections to our current edge, but there was
			// one with more. Use the point we recorded above.
			if (flag)
				return CreateGroup(NormalPoint, record);

			// If we get here there was no connecting next point at all.
			return CreateGroup(NoPoint, record);
		}

		Edge TrackEdge(Mat& edgeFlags, int edgeNo, 
			const unordered_set<Point, PointHash>& junctions,
			const unordered_set<Point, PointHash>& endpoints,
			const Point& start)
		{
			Edge edge;
			edge.Add(start);
			edgeFlags.at<int>(start.y, start.x) = -edgeNo;

			Group<LinkStatus, Point> next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, start);
			LinkStatus status = next.Item1();
			Point nextPoint = next.Item2();

			while (status != NoPoint)
			{
				edge.Add(nextPoint);
				edgeFlags.at<int>(nextPoint.y, nextPoint.x) = -edgeNo;

				if (status == LastPoint) // hit a junction/endpoint
					status = NoPoint;
				else
				{
					next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, nextPoint);
					status = next.Item1();
					nextPoint = next.Item2();
				}
			}

			// Now track from original point in the opposite direction - but only if
			// the starting point was not a junction/endpoint.
			if (junctions.find(start) == junctions.end() && 
				endpoints.find(start) == endpoints.end())
			{
				std::reverse(edge.begin(), edge.end());

				next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, start);
				status = next.Item1();
				nextPoint = next.Item2();

				while (status != NoPoint)
				{
					edge.Add(nextPoint);
					edgeFlags.at<int>(nextPoint.y, nextPoint.x) = -edgeNo;

					if (status == LastPoint)
						status = NoPoint;
					else
					{
						next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, nextPoint);
						status = next.Item1();
						nextPoint = next.Item2();
					}
				}
			}

			// Final check to see if this edge should have start and end points matched 
			// to form a loop. If the number of points in the list is four or more (the 
			// minimum number that could form a loop), and the endpoints are within a pixel 
			// of each other, append a copy if the first point to the end to complete the loop.
			if (edge.Count() >= 4)
			{
				Point start = edge.Front();
				Point end = edge.Back();

				if (abs(start.y - end.y) <= 1 && abs(start.x - end.x) <= 1)
					edge.Add(start);
			}

			return edge;
		}

		// We strongly recommend removing any isolated pixel before applying EdgeLink.
		ArrayList<Edge> EdgeLink(const Mat& binaryImage, int minLength)
		{
			Group<ArrayList<Point>, ArrayList<Point>> juncOrEnd = 
				FindJunctionsOrEndpoints(binaryImage);

			unordered_set<Point, PointHash> junctions, endpoints;
			for (auto p : juncOrEnd.Item1())
				junctions.insert(p);
			for (auto p : juncOrEnd.Item2())
				endpoints.insert(p);

			Mat edgeFlags;
			binaryImage.convertTo(edgeFlags, CV_32S);

			ArrayList<Edge> edgeList;
			for (int i = 0; i < edgeFlags.rows; i++)
			{
				for (int j = 0; j < edgeFlags.cols; j++)
				{
					if (edgeFlags.at<int>(i, j))
					{
						Edge edge = TrackEdge(edgeFlags, (int)edgeList.Count() + 1, 
							junctions, endpoints, Point(j, i));

						if (edge.Count() >= minLength)
							edgeList.Add(edge);
					}
				}
			}

			return edgeList;
		}


		//////////////////////////////////////////////////////////////////////////
		// Others
		//////////////////////////////////////////////////////////////////////////

        Mat GetBoundingBox(const Mat& binaryImage)
        {
            int minX = binaryImage.cols - 1, maxX = 0,
                minY = binaryImage.rows - 1, maxY = 0;

            for (int i = 0; i < binaryImage.rows; i++)
                for (int j = 0; j < binaryImage.cols; j++)
                {
                    if (binaryImage.at<uchar>(i, j))
                    {
                        minX = min(minX, j);
                        maxX = max(maxX, j);
                        minY = min(minY, i);
                        maxY = max(maxY, i);
                    }
                }

                return Mat(binaryImage, Range(minY, maxY + 1), Range(minX, maxX + 1));
        }

		ArrayList<Point> GetEdgels(const Mat& binaryImage)
		{
			ArrayList<Point> points;

			for (int i = 0; i < binaryImage.rows; i++)
				for (int j = 0; j < binaryImage.cols; j++)
					if (binaryImage.at<uchar>(i, j))
						points.Add(Point(j, i));

			return points;
		}

        ArrayList<ArrayList<cv::Point>> GetEdgelChannels(const Mat& binaryImage, int orientNum, 
            double sigma, double lambda)
        {
            ArrayList<cv::Point> points = GetEdgels(binaryImage);
            ArrayList<cv::Mat> gaborChannel = GetGaborChannels(binaryImage, orientNum, sigma, lambda);

            ArrayList<ArrayList<cv::Point>> edgelChannels(orientNum);
            for (int i = 0; i < points.Count(); i++)
            {
                double maxResponse = gaborChannel[0].at<double>(points[i].y, points[i].x);
                int index = 0;

                for (int j = 1; j < orientNum; j++)
                {
                    if (gaborChannel[j].at<double>(points[i].y, points[i].x) > maxResponse)
                    {
                        maxResponse = gaborChannel[j].at<double>(points[i].y, points[i].x);
                        index = j;
                    }
                }

                assert(index >= 0 && index < orientNum);
                edgelChannels[index].Add(points[i]);
            }

            return edgelChannels;
        }

        ArrayList<Point> SampleOnShape(const Mat& binaryImage, size_t samplingNum)
        {
            return SampleFromPoints(GetEdgels(binaryImage), samplingNum);
        }
	}
}
}