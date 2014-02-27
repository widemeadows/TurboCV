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

cv::Mat oraclePreprocess2(const Mat& image)
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

    return binaryImage;
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

    //LocalFeatureCrossValidation<TGabor>("subset", oraclePreprocess);
    Mat image = imread("00003.png", CV_LOAD_IMAGE_GRAYSCALE);
    image = oraclePreprocess(image);
    
    auto edgels = GetEdgels(image);
    Mat X(edgels.Count(), 2, CV_64F);
    cout << edgels.Count() << endl;
    for (int i = 0; i < edgels.Count(); i++)
    {
        X.at<double>(i, 0) = edgels[i].y;
        X.at<double>(i, 1) = edgels[i].x;
    }

    ArrayList<Vec3b> colors(10);
    for (int i = 0; i < colors.Count(); i++)
    {
        colors[i] = Vec3b(rand() / (double)RAND_MAX * 255, 
                          rand() / (double)RAND_MAX * 255, 
                          rand() / (double)RAND_MAX * 255);
    }

    Mat prob = GMM(X, 10).Item1();
    Mat colorImg = Mat::zeros(image.size(), CV_8UC3);
    for (int i = 0; i < edgels.Count(); i++)
    {
        double maxProb = -1;
        int maxIdx = -1;
        for (int j = 0; j < prob.cols; j++)
        if (prob.at<double>(i, j) > maxProb)
        {
            maxProb = prob.at<double>(i, j);
            maxIdx = j;
        }

        colorImg.at<Vec3b>(edgels[i].y, edgels[i].x) = colors[maxIdx];
    }
    imshow("win", colorImg);
    waitKey(0);

    Mat clusterIds = Mat::zeros(edgels.Count(), 1, CV_32S);
    Mat centers(colors.Count(), 2, CV_32F);
    Mat Xf;
    X.convertTo(Xf, CV_32F);
    kmeans(Xf, colors.Count(), clusterIds, TermCriteria(CV_TERMCRIT_ITER, 500, 1e-6), 
        1, KMEANS_PP_CENTERS, centers);
    for (int i = 0; i < edgels.Count(); i++)
    {
        colorImg.at<Vec3b>(edgels[i].y, edgels[i].x) = colors[clusterIds.at<int>(i)];
    }
    imshow("win", colorImg);
    waitKey(0);

    //
    //auto channels = GetGaborChannels(image, 4);
    //for (int i = 0; i < channels.Count(); i++)
    //    blur(channels[i], channels[i], Size(5, 5));
    //for (int i = 0; i < channels.Count(); i++)
    //for (int j = 0; j < image.rows; j++)
    //for (int k = 0; k < image.cols; k++)
    //if (!image.at<uchar>(j, k))
    //    channels[i].at<double>(j, k) = 0.0;

    //for (int j = 0; j < image.rows; j++)
    //for (int k = 0; k < image.cols; k++)
    //if (image.at<uchar>(j, k))
    //{
    //    ArrayList<Group<double, int>> values;
    //    for (int i = 0; i < channels.Count(); i++)
    //        values.Add(CreateGroup(channels[i].at<double>(j, k), i));

    //    sort(values.begin(), values.end());
    //    for (int i = 0; i < values.Count() - 1; i++)
    //        channels[values[i].Item2()].at<double>(j, k) = 0.0;
    //}

    //for (auto channel : channels)
    //{
    //    imshow(channel);
    //    waitKey(0);
    //}
}