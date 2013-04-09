// Export.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Export.h"
#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include <cv.h>
#include <vector>
#include <map>
using namespace System;
using namespace System::Image;
using namespace cv;
using namespace std;

NativeMat::NativeMat(int rows, int cols, int type)
{
    this->rows = rows;
    this->cols = cols;
    this->type = type;

    if (type == EPT_UCHAR)
    {
        m = new uchar*[rows];
        for (int i = 0; i < rows; i++)
            ((uchar**)m)[i] = new uchar[cols];
    }
    else
    {
        m = new float*[rows];
        for (int i = 0; i < rows; i++)
            ((float**)m)[i] = new float[cols];
    }
}

NativeMat::NativeMat(const NativeMat& other)
{
    this->rows = other.rows;
    this->cols = other.cols;
    this->type = other.type;

    if (this->type == EPT_UCHAR)
    {
        m = new uchar*[rows];
        for (int i = 0; i < rows; i++)
        {
            ((uchar**)m)[i] = new uchar[cols];
            for (int j = 0; j < cols; j++)
                ((uchar**)m)[i][j] = ((uchar**)other.m)[i][j];
        }
    }
    else
    {
        m = new float*[rows];
        for (int i = 0; i < rows; i++)
        {
            ((float**)m)[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                ((float**)m)[i][j] = ((float**)other.m)[i][j];
        }
    }
}

NativeMat::~NativeMat()
{
    clear();
}

NativeMat& NativeMat::operator=(const NativeMat& other)
{
    if (m == other.m)
        return *this;

    if (rows != other.rows || cols != other.cols || type != other.type)
    {
        clear();

        rows = other.rows;
        cols = other.cols;
        type = other.type;

        if (this->type == EPT_UCHAR)
        {
            m = new uchar*[rows];
            for (int i = 0; i < rows; i++)
                ((uchar**)m)[i] = new uchar[cols];
        }
        else
        {
            m = new float*[rows];
            for (int i = 0; i < rows; i++)
                ((float**)m)[i] = new float[cols];
        }
    }

    if (this->type == EPT_UCHAR)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ((uchar**)m)[i][j] = ((uchar**)other.m)[i][j];
    }
    else
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ((float**)m)[i][j] = ((float**)other.m)[i][j];
    }

    return *this;
}

void NativeMat::clear()
{
    if (type == EPT_UCHAR)
    {
        for (int i = 0; i < rows; i++)
            delete[] ((uchar**)m)[i];
        delete[] (uchar**)m;
    }
    else
    {
        for (int i = 0; i < rows; i++)
            delete[] ((float**)m)[i];
        delete[] (float**)m;
    }
}

EXPORT_API NativeInfo PerformHitmap(const NativeMat& image, bool thinning)
{
    Hitmap hitmap;

    Mat cvImage(image.rows, image.cols, CV_8U);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            cvImage.at<uchar>(i, j) = image.at<uchar>(i, j);

    Hitmap::Info tmp = hitmap.GetFeatureWithPreprocess(cvImage, thinning);

    NativeInfo result;
    for (int i = 0; i < tmp.size(); i++)
    {
        const Tuple<vector<Point>, Mat>& item = tmp[i];
        const vector<Point>& item1 = item.Item1();
        const Mat& item2 = item.Item2();

        vector<NativePoint> vec;
        for (int j = 0; j < item1.size(); j++)
            vec.push_back(NativePoint(item1[j].x, item1[j].y));

        NativeMat mat(item2.rows, item2.cols, EPT_UCHAR);
        for (int j = 0; j < item2.rows; j++)
            for (int k = 0; k < item2.cols; k++)
                mat.at<uchar>(j, k) = item2.at<uchar>(j, k);

        result.push_back(make_pair(vec, mat));
    }

    return result;
}

EXPORT_API vector<NativeInfo> PerformHitmap(const vector<NativeMat>& images, bool thinning)
{
    vector<NativeInfo> result(images.size());

    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++)
        result[i] = PerformHitmap(images[i], thinning);

    return result;
}