#include "../System/System.h"
#include "Core.h"
#include <algorithm>
#include <numeric>
#include <cv.h>
using namespace std;
using namespace cv;
using namespace TurboCV::System;
using namespace TurboCV::System::Image;


//////////////////////////////////////////////////////////////////////////
// TDTree
//////////////////////////////////////////////////////////////////////////

void TDTree::Build(ArrayList<Point>& points)
{
    nodes = points;
    flags = ArrayList<int>(points.Count());

    innerBuild(nodes.begin(), nodes.end());
}

Group<Point, double> TDTree::Find(const cv::Point& point)
{
    minDist = numeric_limits<double>::max();
    nearest = Point(-1, -1);

    innerFind(nodes.begin(), nodes.end(), point);
    return CreateGroup(nearest, sqrt(minDist));
}
