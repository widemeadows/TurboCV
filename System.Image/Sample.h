#pragma once

#include "../System/System.h"
#include "BinaryImage.h"
#include "Geometry.h"
#include "Util.h"
#include <cv.h>
#include <vector>
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
        inline vector<Point> SampleOnGrid(size_t height, size_t width, size_t samplingNumPerDirection)
        {
            int heightStep = height / samplingNumPerDirection, 
                widthStep = width / samplingNumPerDirection;
            vector<Point> points;

            for (int i = heightStep / 2; i < height; i += heightStep)
                for (int j = widthStep / 2; j < width; j += widthStep)
                    points.push_back(Point(j, i));

            return points;
        }

        inline vector<Point> SampleFromPoints(const vector<Point>& points, size_t samplingNum)
        {
            size_t pointNum = points.size();
	        assert(pointNum >= samplingNum);

            vector<Tuple<double, Tuple<Point, Point>>> distances(pointNum * (pointNum - 1) / 2);
            unordered_set<Point, PointHash> pivots;

            int counter = 0;
            for (size_t i = 0; i < pointNum; i++)
            {
                for (size_t j = i + 1; j < pointNum; j++)
                {
                    double distance = Geometry::EulerDistance(points[i], points[j]);
                    distances[counter++] = CreateTuple(distance, CreateTuple(points[i], points[j]));
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

            vector<Point> results;
            for (auto pivot : pivots)
                results.push_back(pivot);

            return results;
        }

        inline vector<Point> SampleOnShape(const Mat& sketchImage, size_t samplingNum)
        {
            return SampleFromPoints(GetEdgels(sketchImage), samplingNum);
        }
    }
}
}