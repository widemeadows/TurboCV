#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include "MNIST.h"
#include <cv.h>
#include <highgui.h>
#include <random>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;
using namespace cv;
using namespace std;

Descriptor GetBlock(const Mat& image, int top, int left, int blockSize)
{
    Mat block(image, Range(top, top + blockSize), Range(left, left + blockSize));
    Descriptor desc;

    double pixelSum = sum(block)[0];
    if (pixelSum == 0)
    {
        for (int i = 0; i < blockSize; i++)
            for (int j = 0; j < blockSize; j++)
                desc.Add(0);

        return desc;
    }

    ArrayList<Point> points = GetEdgels(block);
    Point center(0, 0);
    for (auto point : points)
    {
        center.x += point.x;
        center.y += point.y;
    }
    center.x /= points.Count();
    center.y /= points.Count();

    int newTop = center.y - blockSize / 2,
        newBottom = newTop + blockSize,
        newLeft = center.x - blockSize / 2,
        newRight = newLeft + blockSize;

    for (int i = newTop; i < newBottom; i++)
    {
        for (int j = newLeft; j < newRight; j++)
        {
            if (i < 0 || i >= block.rows || j < 0 || j >= block.cols)
                desc.Add(0);
            else
                desc.Add(block.at<uchar>(i, j));
        }
    }

    return desc;
}

class TestLevel1: public LocalFeature
{
public:
    TestLevel1(): blockSize(8), descNum(100) {}

    virtual LocalFeatureVec operator()(const cv::Mat& image)
    {
        LocalFeatureVec feature;
        int rowUpperBound = image.rows - blockSize,
            colUpperBound = image.cols - blockSize;
        default_random_engine rowGenerator(100), colGenerator(101);
        uniform_int_distribution<int> rowDistribution(0, rowUpperBound),
            colDistribution(0, colUpperBound);

        for (int i = 0; i < descNum; i++)
        {
            Descriptor desc = GetBlock(image, rowDistribution(rowGenerator), colDistribution(colGenerator), blockSize);
            NormTwoNormalize(desc.begin(), desc.end());

            feature.Add(desc);
        }

        return feature;
    }

    virtual TString GetName() const
    {
        return "level1";
    }

private:
    int blockSize, descNum;
};

class TestLevel2
{
public:
    TestLevel2(): blockSize(8) {}

    virtual ArrayList<LocalFeatureVec> operator()(const cv::Mat& image)
    {
        ArrayList<LocalFeatureVec> features(4);
        
        for (int i = 0; i < image.rows - blockSize; i++)
        {
            for (int j = 0; j < image.cols - blockSize; j++)
            {
                Descriptor desc = GetBlock(image, i, j, blockSize);
                NormTwoNormalize(desc.begin(), desc.end());

                if (i < (image.rows - blockSize) / 2)
                {
                    if (j < (image.cols - blockSize) / 2)
                        features[0].Add(desc);
                    else
                        features[1].Add(desc);
                }
                else
                {
                    if (j < (image.cols - blockSize) / 2)
                        features[2].Add(desc);
                    else
                        features[3].Add(desc);
                }
            }
        }

        return features;
    }

    virtual TString GetName() const
    {
        return "level2";
    }

private:
    int blockSize;
};

inline ArrayList<Histogram> test(const TString& datasetPath)
{
    ArrayList<TString> imagePaths = Solver::LoadDataset(datasetPath).Item1();
    int nImage = imagePaths.Count();
    ArrayList<Mat> images(nImage);

    #pragma omp parallel for
    for (int i = 0; i < nImage; i++)
    {
        Mat image = imread(imagePaths[i], CV_LOAD_IMAGE_GRAYSCALE);
        image = reverse(image);
        resize(image, image, Size(24, 24), 0, 0, CV_INTER_AREA);
        //blur(image, image, Size(2, 2));
        //threshold(image, image, 0, 255, CV_THRESH_BINARY);

        images[i] = image;
    }

    ArrayList<Descriptor_f> allDesc;
    for (int i = 0; i < nImage; i++)
    {
        LocalFeatureVec_f feature;
        Convert(TestLevel1()(images[i]), feature);

        for (Descriptor_f& desc : feature)
            allDesc.Add(desc);
    }

    allDesc = PCAWhiten(allDesc);
    ArrayList<Word_f> words = BOV(allDesc, 1000).GetVisualWords();

    ArrayList<LocalFeatureVec_f> featureList[4];
    for (int i = 0; i < nImage; i++)
    {
        ArrayList<LocalFeatureVec> features = TestLevel2()(images[i]);

        for (int j = 0; j < 4; j++)
        {
            LocalFeatureVec_f feature;
            Convert(features[j], feature);
            featureList[j].Add(feature);
        }
    }

    printf("Frequency Histogram...\n");
    ArrayList<Histogram> histograms[4];
    for (int i = 0; i < 4; i++)
    {
        histograms[i] = FreqHist(featureList[i], words).GetFrequencyHistograms();
        for (int j = 0; j < histograms[i].Count(); j++)
            for (int k = 0; k < histograms[i][j].Count(); k++)
                histograms[i][j][k] /= 4.0;
    }

    ArrayList<Histogram> finalFeatures(histograms[0].Count());
    for (int i = 0; i < finalFeatures.Count(); i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < histograms[j][i].Count(); k++)
                finalFeatures[i].Add(histograms[j][i][k]);

    printf("%d\n", (int)finalFeatures[0].Count());
    return finalFeatures;
}