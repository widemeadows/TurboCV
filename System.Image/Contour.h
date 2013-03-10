#pragma once

#include "Util.h"
#include "Geometry.h"
#include <cv.h>
#include <tuple>
#include <unordered_set>
#include <algorithm>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        inline vector<Point> GetPivots(const vector<Point>& points, size_t pivotNum)
        {
            size_t pointNum = points.size();
	        assert(pointNum >= pivotNum);

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
	        while (pivots.size() > pivotNum)
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
    }
}