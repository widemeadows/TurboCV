#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include "../System.XML/System.XML.h"
#include "CrossValidation.hpp"
#include "Util.h"
#include <cv.h>
#include <highgui.h>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;
using namespace cv;
using namespace std;

#define SAVE_FEATURE
//#define SAVE_DISTANCE_MATRIX

//////////////////////////////////////////////////////////////////////////
// Preprocess
//////////////////////////////////////////////////////////////////////////

cv::Mat sketchPreprocess(const cv::Mat& image)
{
    cv::Mat finalImage = reverse(image);
    cv::resize(finalImage, finalImage, cv::Size(256, 256));

    return finalImage;
}

cv::Mat mayaPreprocess(const cv::Mat& image)
{
    int leftPadding = 0, rightPadding = 0, topPadding = 0, bottomPadding = 0;
    if (image.rows < image.cols)
    {
        topPadding = (image.cols - image.rows) / 2;
        bottomPadding = 512 - image.rows - topPadding;
    }
    else
    {
        leftPadding = (image.rows - image.cols) / 2;
        rightPadding = 512 - image.cols - leftPadding;
    }

    cv::Mat squareImage;
    copyMakeBorder(image, squareImage, topPadding, bottomPadding, leftPadding, rightPadding, 
        BORDER_CONSTANT, Scalar(0, 0, 0, 0));
    assert(squareImage.rows == 512 && squareImage.cols == 512);

    cv::Mat scaledImage;
    resize(squareImage, scaledImage, Size(256, 256), 0.0, 0.0, CV_INTER_AREA);

    cv::Mat finalImage;
    threshold(scaledImage, finalImage, 54, 255, CV_THRESH_BINARY);

    return finalImage;
}

cv::Mat oraclePreprocess(const Mat& image)
{
    Mat boundingBox = GetBoundingBox(reverse(image));

    Mat squareImage;
    int leftPadding = 0, topPadding = 0;
    if (boundingBox.rows < boundingBox.cols)
        topPadding = (boundingBox.cols - boundingBox.rows) / 2;
    else
        leftPadding = (boundingBox.rows - boundingBox.cols) / 2;
    copyMakeBorder(boundingBox, squareImage, topPadding, topPadding, 
        leftPadding, leftPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

    Mat scaledImage;
    resize(squareImage, scaledImage, Size(228, 228));

    Mat paddedImage;
    copyMakeBorder(scaledImage, paddedImage, 14, 14, 14, 14, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
    assert(paddedImage.rows == 256 && paddedImage.cols == 256);

    Mat binaryImage;
    threshold(paddedImage, binaryImage, 200, 255, CV_THRESH_BINARY);

    Mat finalImage;
    thin(binaryImage, finalImage);

    return finalImage;
}

void Batch(const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    LocalFeatureCrossValidation<HOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RHOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<SHOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<LogSHOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<HOOSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RHOOSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<SC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<PSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RPSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RGabor>(datasetPath, preprocess);
    printf("\n");

    GlobalFeatureCrossValidation<GHOG>(datasetPath, preprocess);
    printf("\n");

    GlobalFeatureCrossValidation<GIST>(datasetPath, preprocess);
    printf("\n");

    EdgeMatchCrossValidation<CM>(datasetPath, preprocess);
    printf("\n");

    EdgeMatchCrossValidation<OCM>(datasetPath, preprocess);
    printf("\n");

    EdgeMatchCrossValidation<Hitmap>(datasetPath, preprocess);
    printf("\n");
}

void Choose(const TString& algoName, const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    if (algoName == TString("hog"))
    {
        LocalFeatureCrossValidation<HOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rhog"))
    {
        LocalFeatureCrossValidation<RHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("shog"))
    {
        LocalFeatureCrossValidation<SHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("logshog"))
    {
        LocalFeatureCrossValidation<LogSHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("hoosc"))
    {
        LocalFeatureCrossValidation<HOOSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rhoosc"))
    {
        LocalFeatureCrossValidation<RHOOSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("sc"))
    {
        LocalFeatureCrossValidation<SC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("psc"))
    {
        LocalFeatureCrossValidation<PSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rsc"))
    {
        LocalFeatureCrossValidation<RSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rpsc"))
    {
        LocalFeatureCrossValidation<RPSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rgabor"))
    {
        LocalFeatureCrossValidation<RGabor>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("ghog"))
    {
        GlobalFeatureCrossValidation<GHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("gist"))
    {
        GlobalFeatureCrossValidation<GIST>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("cm"))
    {
        EdgeMatchCrossValidation<CM>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("ocm"))
    {
        EdgeMatchCrossValidation<OCM>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("hitmap"))
    {
        EdgeMatchCrossValidation<Hitmap>(datasetPath, preprocess);
        printf("\n");
    }
}

Group<ArrayList<Word_f>, ArrayList<Histogram>, ArrayList<int>> LoadLocalFeatureData(const TString& fileName)
{
    FILE* file = fopen(fileName, "r");
    int nRow, nCol;

    fscanf(file, "%d %d", &nRow, &nCol);

    ArrayList<Word_f> words(nRow);

    for (int i = 0; i < nRow; i++)
    {
        Word_f word(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &word[j]);

        words[i] = word;
    }

    fscanf(file, "%d %d", &nRow, &nCol);

    ArrayList<int> labels(nRow);
    ArrayList<Histogram> histograms(nRow);

    for (int i = 0; i < nRow; i++)
    {
        fscanf(file, "%d", &labels[i]);

        Histogram histogram(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &histogram[j]);

        histograms[i] = histogram;
    }

    fclose(file);

    return CreateGroup(words, histograms, labels);
}

Group<ArrayList<PointList>, ArrayList<Mat>> GetBlock(
    int x, int y, size_t blockSize,
    const ArrayList<PointList>& pointLists)
{
    int expectedTop = (int)(y - blockSize / 2),
        expectedLeft = (int)(x - blockSize / 2),
        expectedBottom = expectedTop + blockSize,
        expectedRight = expectedLeft + blockSize;

    ArrayList<PointList> pLists;

    for (int i = 0; i < pointLists.Count(); i++)
    {
        PointList pList;

        for (int j = 0; j < pointLists[i].Count(); j++)
        {
            if (expectedLeft <= pointLists[i][j].x && pointLists[i][j].x < expectedRight &&
                expectedTop <= pointLists[i][j].y && pointLists[i][j].y < expectedBottom)
                pList.Add(Point(pointLists[i][j].x - expectedLeft, 
                                pointLists[i][j].y - expectedTop));
        }

        pLists.Add(pList);
    }

    OCM machine;
    return CreateGroup(pLists, machine.GetTransforms(Size(blockSize, blockSize), pLists));
}

double GetOneWayDistance(const ArrayList<PointList>& u, ArrayList<TDTree>& v)
{
    int orientNum = u.Count(), uPointNum = 0;
    double uToV = 0;

    for (int i = 0; i < orientNum; i++)
    {
        const ArrayList<Point>& uPoints = u[i];
        TDTree& vTree = v[i];

        for (size_t i = 0; i < uPoints.Count(); i++)
            uToV += vTree.Find(uPoints[i]).Item2();

        uPointNum += uPoints.Count();
    }

    if (uPointNum == 0)
        return numeric_limits<double>::max();
    else
        return uToV / uPointNum;
}

double GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v)
{
    assert(u.Count() == v.Count());
    int orientNum = u.Count(), uPointNum = 0;
    double uToV = 0;

    for (int i = 0; i < orientNum; i++)
    {
        const ArrayList<Point>& uPoints = u[i];
        const Mat& vMat = v[i];

        for (size_t i = 0; i < uPoints.Count(); i++)
            uToV += sqrt(vMat.at<float>(uPoints[i].y, uPoints[i].x));

        uPointNum += uPoints.Count();
    }

    if (uPointNum == 0)
        return numeric_limits<double>::max();
    else
        return uToV / uPointNum;
}

int main(int argc, char* argv[])
{
    //EnLocalFeatureCrossValidation<RGabor>("sketches", sketchPreprocess);
    //EnLocalFeatureCrossValidation<RGabor>("oracles", oraclePreprocess);
    //LocalFeatureCrossValidation<RGabor>("subset", sketchPreprocess);
    //LocalFeatureCrossValidation<RHOG>("oracles", oraclePreprocess);

    LocalFeatureCrossValidation<TGabor>("subset", sketchPreprocess);

    //auto dataset = LoadDataset("subset");
    //auto paths = dataset.Item1();
    //auto labels = dataset.Item2();

    //int nImage = paths.Count();

    //printf("ImageNum: %d\n", nImage);
    //printf("Cross Validation on Local Hitmap...\n");
    //
    //ArrayList<cv::Mat> images(nImage);
    //ArrayList<ArrayList<PointList>> channels(nImage);
    //#pragma omp parallel for
    //for (int i = 0; i < nImage; i++)
    //{
    //    cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE);
    //    images[i] = sketchPreprocess(image);
    //    channels[i] = GetEdgelChannels(images[i], 4);
    //}

    //cout << "Clustering" << endl;
    //ArrayList<Point> points = SampleOnGrid(images[0].rows, images[0].cols, 14);
    //size_t blockSize = 92, clusterNum = 500;

    //ArrayList<Group<int, int, int>> tmpPoints;
    //for (int i = 0; i < nImage; i++)
    //for (size_t j = 0; j < points.Count(); j++)
    //    tmpPoints.Add(CreateGroup(i, points[j].x, points[j].y));
    //ArrayList<Group<int, int, int>> centerPoints = RandomPickUp(tmpPoints, clusterNum);

    //ArrayList<Group<ArrayList<PointList>, ArrayList<Mat>>> centers(clusterNum);
    //#pragma omp parallel for
    //for (int i = 0; i < clusterNum; i++)
    //{
    //    centers[i] = GetBlock(centerPoints[i].Item2(), centerPoints[i].Item3(), blockSize,
    //        channels[centerPoints[i].Item1()]);
    //}

    //cout << "Bagging" << endl;
    //ArrayList<Histogram> histograms(nImage);
    //#pragma omp parallel for
    //for (int i = 0; i < nImage; i++)
    //{
    //    Histogram hist(clusterNum);

    //    for (int j = 0; j < points.Count(); j++)
    //    {
    //        Group<ArrayList<PointList>, ArrayList<Mat>> block = GetBlock(
    //            points[j].x, points[j].y, blockSize, channels[i]);

    //        for (int k = 0; k < clusterNum; k++)
    //        {
    //            double d1 = GetOneWayDistance(block.Item1(), centers[k].Item2());
    //            double d2 = GetOneWayDistance(centers[k].Item1(), block.Item2());
    //            //d1 = min(d1, 40.0);
    //            //d2 = min(d2, 40.0);
    //            double distance = (d1 + d2) / 2.0;

    //            hist[k] += Math::Gauss(distance, 2.0);
    //        }
    //    }

    //    NormOneNormalize(hist.begin(), hist.end());
    //    //for (int j = 0; j < hist.Count(); j++)
    //    //    hist[j] /= points.Count();
    //    histograms[i] = hist;
    //    printf("%d\n", i);
    //}

    //auto evaIdxes = SplitDatasetRandomly(labels, 3);
    //for (int i = 0; i < evaIdxes.Count(); i++)
    //{
    //    printf("\nBegin Fold %d...\n", i + 1);
    //    const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
    //    ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
    //    ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
    //    ArrayList<Histogram> trainingHistograms = Divide(histograms, pickUpIndexes).Item2();
    //    ArrayList<Histogram> evaluationHistograms = Divide(histograms, pickUpIndexes).Item1();

    //    printf("Fold %d Accuracy: ", i + 1);
    //    cout << KNN<Histogram>().
    //        Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).Item1() << endl;
    //}

    //ArrayList<GlobalFeatureVec_f> fHists;
    //for (int i = 0; i < histograms.Count(); i++)
    //{
    //    GlobalFeatureVec_f tmp;
    //    Convert(histograms[i], tmp);
    //    fHists.Add(tmp);
    //}
    //SaveGlobalFeatures("test_data", fHists, labels);
}