// Export.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Export.h"
#include "../System.Image/System.Image.h"
#include <cv.h>
#include <vector>
#include <map>
using namespace System::Image;
using namespace cv;
using namespace std;

EXPORT_API vector<pair<vector<Position>, Matrix<float>>> PerformHitmap(
    const Matrix<uchar>& image, bool thinning)
{
    Hitmap hitmap;

    Mat cvImage(image.rows, image.cols, CV_8U);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            cvImage.at<uchar>(i, j) = image.at(i, j);

    Hitmap::Info tmp = hitmap.GetFeatureWithPreprocess(cvImage, thinning);

    vector<pair<vector<Position>, Matrix<float>>> result;
    for (int i = 0; i < tmp.size(); i++)
    {
        const Tuple<vector<Point>, Mat>& item = tmp[i];
        const vector<Point>& item1 = item.Item1();
        const Mat& item2 = item.Item2();

        vector<Position> vec;
        for (int j = 0; j < item1.size(); j++)
            vec.push_back(Position(item1[j].x, item1[j].y));
    
        Matrix<float> mat(item2.rows, item2.cols);
        for (int j = 0; j < item2.rows; j++)
            for (int k = 0; k < item2.cols; k++)
                mat.at(j, k) = item2.at<float>(j, k);

        result.push_back(make_pair(vec, mat));
    }

    return result;
}