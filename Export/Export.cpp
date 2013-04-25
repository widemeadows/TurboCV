// Export.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Export.h"
#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include <cv.h>
#include <vector>
#include <map>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace cv;
using namespace std;

NativeMat::NativeMat(int rows, int cols, BasicType type)
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

template<typename T>
Mat ConvertNativeMatToCvMat(const NativeMat& mat, T type)
{
    Mat cvMat(mat.rows, mat.cols, CV_8U);

    for (int i = 0; i < mat.rows; i++)
        for (int j = 0; j < mat.cols; j++)
            cvMat.at<T>(i, j) = mat.at<T>(i, j);

    return cvMat;
};

EXPORT_API NativeInfo EdgeMatchingPredict(EdgeMatchingType type,
    const NativeMat& image, bool thinning)
{
    Mat cvImage = ConvertNativeMatToCvMat(image, uchar());

    TurboCV::System::ArrayList<Tuple<vector<Point>, Mat>> tmp;
    if (type == EPT_OCM)
        tmp = OCM().GetFeatureWithPreprocess(cvImage, thinning);
    else
        tmp = Hitmap().GetFeatureWithPreprocess(cvImage, thinning);
        
    NativeInfo result;
    for (int i = 0; i < tmp.Count(); i++)
    {
        const Tuple<vector<Point>, Mat>& item = tmp[i];
        const vector<Point>& item1 = item.Item1();
        const Mat& item2 = item.Item2();

        vector<NativePoint> vec;
        for (int j = 0; j < item1.size(); j++)
            vec.push_back(NativePoint(item1[j].x, item1[j].y));

        if (type == EPT_OCM)
        {
            NativeMat mat(item2.rows, item2.cols, EPT_FLOAT);
            for (int j = 0; j < item2.rows; j++)
                for (int k = 0; k < item2.cols; k++)
                    mat.at<float>(j, k) = item2.at<float>(j, k);

            result.push_back(make_pair(vec, mat));
        }
        else
        {
            NativeMat mat(item2.rows, item2.cols, EPT_UCHAR);
            for (int j = 0; j < item2.rows; j++)
                for (int k = 0; k < item2.cols; k++)
                    mat.at<uchar>(j, k) = item2.at<uchar>(j, k);

            result.push_back(make_pair(vec, mat));
        }
    }

    return result;
}

EXPORT_API vector<NativeInfo> EdgeMatchingPredict(EdgeMatchingType type, 
    const vector<NativeMat>& images, bool thinning)
{
    vector<NativeInfo> result(images.size());

    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++)
        result[i] = EdgeMatchingPredict(type, images[i], thinning);

    return result;
}

EXPORT_API pair<vector<NativeWord>, vector<NativeHistogram>> LocalFeatureTrain(LocalFeatureType type,
    const vector<NativeMat>& images, int wordNum, bool thinning)
{
    int imageNum = images.size();
    vector<LocalFeature_f> features(imageNum);
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
    {
        Mat cvImage = ConvertNativeMatToCvMat(images[i], uchar());

        switch (type)
        {
        case EPT_RHOG:
            Convert(RHOG().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        default:
            break;
        }
    }

    vector<Word_f> words = BOV::GetVisualWords(features, wordNum, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    vector<NativeWord> nativeWords(words.size());
    for (int i = 0; i < words.size(); i++)
        for (int j = 0; j < words[i].Count(); j++)
            nativeWords[i].push_back(words[i][j]);

    vector<NativeHistogram> nativeHistograms(freqHistograms.size());
    for (int i = 0; i < freqHistograms.size(); i++)
        for (int j = 0; j < freqHistograms[i].Count(); j++)
            nativeHistograms[i].push_back(freqHistograms[i][j]);

    return make_pair(nativeWords, nativeHistograms);
}

EXPORT_API NativeHistogram LocalFeaturePredict(LocalFeatureType type, const NativeMat& image, 
    const vector<NativeWord>& words, bool thinning)
{
    LocalFeature_f feature;
    Mat cvImage = ConvertNativeMatToCvMat(image, uchar());

    switch (type)
    {
    case EPT_RHOG:
        Convert(RHOG().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    default:
        break;
    }

    vector<Word_f> cvWords(words.size());
    for (int i = 0; i < words.size(); i++)
        for (int j = 0; j < words[i].size(); j++)
            cvWords[i].Add(words[i][j]);
    Histogram freqHistogram = BOV::GetFrequencyHistogram(feature, cvWords);

    NativeHistogram nativeHistogram;
    for (int i = 0; i < freqHistogram.Count(); i++)
        nativeHistogram.push_back(freqHistogram[i]);

    return nativeHistogram;
}

EXPORT_API vector<NativeHistogram> LocalFeaturePredict(LocalFeatureType type, const vector<NativeMat>& images, 
    const vector<NativeWord>& words, bool thinning)
{
    vector<NativeHistogram> histograms(images.size());

    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++)
        histograms[i] = LocalFeaturePredict(type, images[i], words, thinning);

    return histograms;
}