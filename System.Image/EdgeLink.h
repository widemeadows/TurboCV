#pragma once

#include "../System/System.h"
#include "Util.h"
#include <cv.h>
#include <algorithm>
#include <unordered_set>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        typedef ArrayList<Point> Edge;

        enum Status
        {
            NormalPoint,
            LastPoint,
            NoPoint,
        };

        // Assume: Background is 0.
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

        // Assume: Edgels are 1 and Background is 0.
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
        // Assume: Edgels are 1 and Background is 0.
        Group<Status, Point> NextPoint(const Mat& edgeFlags, int edgeNo, 
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
                    if (edgeFlags.at<int>(newY, newX) == 1)
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

            Group<Status, Point> next = NextPoint(edgeFlags, edgeNo, junctions, endpoints, start);
            Status status = next.Item1();
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

        // Assume: Edgels are 1 and Background is 0. 
        // We strongly recommend removing any isolated pixel before applying EdgeLink.
        ArrayList<Edge> EdgeLink(const Mat& binaryImage, int minLength = 10)
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
                    if (edgeFlags.at<int>(i, j) == 1)
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
    }
}
}