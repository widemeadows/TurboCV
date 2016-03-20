// Export.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Export.h"
#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include <cv.h>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;

template<typename T>
cv::Mat ConvertNativeMatToCvMat(const NativeMat& mat, T type)
{
    cv::Mat cvMat(mat.rows, mat.cols, CV_8U);

    for (int i = 0; i < mat.rows; i++)
        for (int j = 0; j < mat.cols; j++)
            cvMat.at<T>(i, j) = mat.at<T>(i, j);

    return cvMat;
};

EXPORT_API NativeInfo EdgeMatchingPredict(EdgeMatchingType type,
    const NativeMat& image, bool thinning)
{
    cv::Mat cvImage = ConvertNativeMatToCvMat(image, uchar());

    ArrayList<Group<ArrayList<cv::Point>, cv::Mat>> tmp;
    if (type == EPT_OCM)
        tmp = OCM().GetFeatureWithPreprocess(cvImage, thinning);
    else
        tmp = Hitmap().GetFeatureWithPreprocess(cvImage, thinning);

    NativeInfo result;
    for (int i = 0; i < tmp.Count(); i++)
    {
        const Group<ArrayList<cv::Point>, cv::Mat>& item = tmp[i];
        const ArrayList<cv::Point>& item1 = item.Item1();
        const cv::Mat& item2 = item.Item2();

        ArrayList<NativePoint> points;
        for (int j = 0; j < item1.Count(); j++)
            points.Add(NativePoint(item1[j].x, item1[j].y));

        if (type == EPT_OCM)
        {
            NativeMat mat(item2.rows, item2.cols, EPT_FLOAT);
            for (int j = 0; j < item2.rows; j++)
                for (int k = 0; k < item2.cols; k++)
                    mat.at<float>(j, k) = item2.at<float>(j, k);

            result.Add(CreateGroup(points, mat));
        }
        else
        {
            NativeMat mat(item2.rows, item2.cols, EPT_UCHAR);
            for (int j = 0; j < item2.rows; j++)
                for (int k = 0; k < item2.cols; k++)
                    mat.at<uchar>(j, k) = item2.at<uchar>(j, k);

            result.Add(CreateGroup(points, mat));
        }
    }

    return result;
}

EXPORT_API ArrayList<NativeInfo> EdgeMatchingPredict(EdgeMatchingType type, 
    const ArrayList<NativeMat>& images, bool thinning)
{
    ArrayList<NativeInfo> result(images.Count());

    #pragma omp parallel for
    for (int i = 0; i < images.Count(); i++)
        result[i] = EdgeMatchingPredict(type, images[i], thinning);

    return result;
}

EXPORT_API Group<ArrayList<NativeWord>, ArrayList<NativeHistogram>> LocalFeatureTrain(
    LocalFeatureType type,
    const ArrayList<NativeMat>& images, 
    int wordNum, 
    bool thinning)
{
    int imageNum = images.Count();
    ArrayList<LocalFeatureVec_f> features(imageNum);

    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
    {
        cv::Mat cvImage = ConvertNativeMatToCvMat(images[i], uchar());

        switch (type)
        {
        case EPT_RHOG:
            Convert(RHOG().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        case EPT_SC:
            Convert(SC().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        case EPT_RSC:
            Convert(RSC().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        case EPT_PSC:
            Convert(PSC().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        case EPT_RPSC:
            Convert(RPSC().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        case EPT_HOOSC:
            Convert(HOOSC().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        case EPT_RHOOSC:
            Convert(RHOOSC().GetFeatureWithPreprocess(cvImage, thinning), features[i]);
            break;
        default:
            break;
        }
    }

    ArrayList<NativeWord> words = BOV::GetVisualWords(features, wordNum, 1000000);
    ArrayList<NativeHistogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    return CreateGroup(words, freqHistograms);
}

EXPORT_API NativeHistogram LocalFeaturePredict(
    LocalFeatureType type, 
    const NativeMat& image, 
    const ArrayList<NativeWord>& words, 
    bool thinning)
{
    LocalFeatureVec_f feature;
    cv::Mat cvImage = ConvertNativeMatToCvMat(image, uchar());

    switch (type)
    {
    case EPT_RHOG:
        Convert(RHOG().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    case EPT_SC:
        Convert(SC().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    case EPT_RSC:
        Convert(RSC().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    case EPT_PSC:
        Convert(PSC().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    case EPT_RPSC:
        Convert(RPSC().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    case EPT_HOOSC:
        Convert(HOOSC().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    case EPT_RHOOSC:
        Convert(RHOOSC().GetFeatureWithPreprocess(cvImage, thinning), feature);
        break;
    default:
        break;
    }

    return BOV::GetFrequencyHistogram(feature, words);
}

EXPORT_API ArrayList<NativeHistogram> LocalFeaturePredict(
    LocalFeatureType type, 
    const ArrayList<NativeMat>& images, 
    const ArrayList<NativeWord>& words, 
    bool thinning)
{
    ArrayList<NativeHistogram> histograms(images.Count());

    #pragma omp parallel for
    for (int i = 0; i < images.Count(); i++)
        histograms[i] = LocalFeaturePredict(type, images[i], words, thinning);

    return histograms;
}

EXPORT_API NativeVec GlobalFeaturePredict(
	GlobalFeatureType type, 
	const NativeMat& image, 
	bool thinning)
{
	GlobalFeatureVec feature;
	cv::Mat cvImage = ConvertNativeMatToCvMat(image, uchar());

	switch (type)
	{
	case EPT_GHOG:
		feature = GHOG().GetFeatureWithPreprocess(cvImage, thinning);
		break;
	case EPT_GIST:
		feature = GIST().GetFeatureWithPreprocess(cvImage, thinning);
		break;
	default:
		break;
	}

	return feature;
}

EXPORT_API ArrayList<NativeVec> GlobalFeaturePredict(
	GlobalFeatureType type, 
	const ArrayList<NativeMat>& images, 
	bool thinning)
{
	ArrayList<NativeVec> vecs(images.Count());

	#pragma omp parallel for
	for (int i = 0; i < images.Count(); i++)
		vecs[i] = GlobalFeaturePredict(type, images[i], thinning);

	return vecs;
}