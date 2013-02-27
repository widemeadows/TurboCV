#pragma once

#include "Util.h"
#include "Geometry.h"
#include <cv.h>
#include <unordered_set>
#include <algorithm>
using namespace cv;
using namespace std;

namespace System
{
    namespace Image
    {
        class Contour
        {
        public:
            static vector<Point> GetEdgels(const Mat& sketchImage);
            
            static bool Compare(const tuple<double, tuple<Point, Point>>& u, 
                const tuple<double, tuple<Point, Point>>& v);

            static vector<Point> GetPivots(const vector<Point>& points, int pivotNum);
        };

        inline bool Contour::Compare(const tuple<double, tuple<Point, Point>>& u, 
            const tuple<double, tuple<Point, Point>>& v)
        {
            return u < v;
        }

        inline vector<Point> Contour::GetEdgels(const Mat& sketchImage)
        {
	        vector<Point> points;

	        for (int i = 0; i < sketchImage.rows; i++)
		        for (int j = 0; j < sketchImage.cols; j++)
			        if (sketchImage.at<uchar>(i, j))
				        points.push_back(Point(j, i));

	        return points;
        }

        inline vector<Point> Contour::GetPivots(const vector<Point>& points, int pivotNum)
        {
            int pointNum = points.size();
	        assert(pointNum >= pivotNum);

	        vector<tuple<double, tuple<Point, Point>>> distances;
	        unordered_set<Point, PointHash> pivots;

	        for (int i = 0; i < pointNum; i++)
	        {
		        for (int j = i + 1; j < pointNum; j++)
		        {
			        double distance = Geometry::EulerDistance(points[i], points[j]);
			        distances.push_back(make_tuple(distance, make_tuple(points[i], points[j])));
		        }
		        pivots.insert(points[i]);
	        }
	        sort(distances.begin(), distances.end(), Contour::Compare);

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