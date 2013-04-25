#pragma once

#include "../System/System.h"
#include "BinaryImage.h"
#include "Geometry.h"
#include "Util.h"
#include <cv.h>
#include <unordered_set>
#include <algorithm>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
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
            unordered_set<Point, PointHash> pivots;

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