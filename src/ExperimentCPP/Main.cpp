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

    fscanf(file, "%nDesc %nDesc", &nRow, &nCol);

    ArrayList<Word_f> words(nRow);

    for (int i = 0; i < nRow; i++)
    {
        Word_f word(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &word[j]);

        words[i] = word;
    }

    fscanf(file, "%nDesc %nDesc", &nRow, &nCol);

    ArrayList<int> labels(nRow);
    ArrayList<Histogram> histograms(nRow);

    for (int i = 0; i < nRow; i++)
    {
        fscanf(file, "%nDesc", &labels[i]);

        Histogram histogram(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &histogram[j]);

        histograms[i] = histogram;
    }

    fclose(file);

    return CreateGroup(words, histograms, labels);
}

int main(int argc, char* argv[])
{
    //EnLocalFeatureCrossValidation<RGabor>("oracles", oraclePreprocess);
    LocalFeatureCrossValidation<RHOG>("sketches", sketchPreprocess);

    /*EnLocalFeatureSolver<RGabor> solver(oraclePreprocess, "subset");
    solver.CrossValidation();

    auto labels = solver.GetLabels();
    auto histograms = solver.GetHistograms();

    FILE* file = fopen("rgabor_hist.txt", "w");

    fprintf(file, "%d %d\n", histograms.Count(), histograms[0].Count());
    for (int i = 0; i < histograms.Count(); i++)
    {
        for (int j = 0; j < histograms[i].Count(); j++)
            fprintf(file, "%f ", histograms[i][j]);

        fprintf(file, "\n");
    }

    fclose(file);

    cv::Mat result = TSNE::Compute(histograms);

    file = fopen("Y.txt", "w");
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
            fprintf(file, "%f ", result.at<double>(i, j));
        fprintf(file, "\n");
    }

    fclose(file);

    file = fopen("labels.txt", "w");

    for (int i = 0; i < labels.Count(); i++)
        fprintf(file, "%d\n", labels[i]);
    
    fclose(file);*/
}