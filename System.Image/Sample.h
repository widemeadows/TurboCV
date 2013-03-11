#pragma once

#include "BinaryImage.h"
#include "Geometry.h"
#include "Util.h"
#include <cv.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
using namespace cv;
using namespace std;

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

	        vector<tuple<double, tuple<Point, Point>>> distances;
	        unordered_set<Point, PointHash> pivots;

	        for (size_t i = 0; i < pointNum; i++)
	        {
		        for (size_t j = i + 1; j < pointNum; j++)
		        {
			        double distance = Geometry::EulerDistance(points[i], points[j]);
			        distances.push_back(make_tuple(distance, make_tuple(points[i], points[j])));
		        }
		        pivots.insert(points[i]);
	        }
	        sort(distances.begin(), distances.end(), [](
                const tuple<double, tuple<Point, Point>>& u, 
                const tuple<double, tuple<Point, Point>>& v)
                { return u < v; });

	        int ptr = 0;
	        while (pivots.size() > samplingNum)
	        {
		        tuple<Point, Point> pointPair = get<1>(distances[ptr++]);
		        if (pivots.find(get<0>(pointPair)) != pivots.end() &&
			        pivots.find(get<1>(pointPair)) != pivots.end())
		        pivots.erase(get<1>(pointPair));
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