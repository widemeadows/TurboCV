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

    return squareImage;
}

//Mat Preprocess(const Mat& sketchImage)
//{
//    Mat binaryImage;
//    threshold(sketchImage, binaryImage, 200, 1, CV_THRESH_BINARY_INV);
//
//    Mat boundingBox = GetBoundingBox(binaryImage);
//
//    Mat squareImage;
//    int leftPadding = 0, topPadding = 0;
//    if (boundingBox.rows < boundingBox.cols)
//        topPadding = (boundingBox.cols - boundingBox.rows) / 2;
//    else
//        leftPadding = (boundingBox.rows - boundingBox.cols) / 2;
//    copyMakeBorder(boundingBox, squareImage, topPadding, topPadding, 
//        leftPadding, leftPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
//
//    Mat scaledImage;
//    resize(squareImage, scaledImage, Size(228, 228));
//
//    Mat paddedImage;
//    copyMakeBorder(scaledImage, paddedImage, 14, 14, 14, 14, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
//    assert(paddedImage.rows == size.height && paddedImage.cols == size.width);
//
//    Mat finalImage;
//    thin(paddedImage, finalImage);
//
//    return finalImage;
//}

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
    if (argc == 2)
        Choose(argv[1], "subset", sketchPreprocess);

    //ArrayList<LocalFeatureVec_f> features;
    //double tmp;

    //DirectoryInfo dir("features");
    //ArrayList<TString> fileNames = dir.GetFiles();

    //for (int i = 0; i < fileNames.Count(); i++)
    //{
    //    LocalFeatureVec_f localFeature;
    //    
    //    FILE* file = fopen(fileNames[i], "r");
    //    while (fscanf(file, "%lf", &tmp) == 1)
    //    {
    //        Descriptor_f desc;

    //        desc.Add(tmp);
    //        for (int j = 1; j < 480; j++)
    //        {
    //            fgetc(file);
    //            fscanf(file, "%lf", &tmp);
    //            desc.Add(tmp);
    //        }

    //        localFeature.Add(desc);
    //    }
    //    fclose(file);

    //    features.Add(localFeature);
    //}

    //ArrayList<TString> paths = LoadDataset("mayas").Item1();
    //ArrayList<int> labels = LoadDataset("mayas").Item2();
    //std::map<TString, TString> params;
    //int nImage = paths.Count();

    //params["angleNum"] = "12";
    //params["orientNum"] = "8";
    //params["pivotRatio"] = "0.1";
    //HOOSC(params, true); // display all params of the algorithm

    //ArrayList<LocalFeatureVec_f> features(nImage);

    //#pragma omp parallel for
    //for (int i = 0; i < nImage; i++)
    //{
    //    HOOSC machine(params);
    //    cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE); 
    //    Convert(machine(mayaPreprocess != NULL ? mayaPreprocess(image) : image), features[i]);
    //}

    //printf("Compute Visual Words...\n");
    //ArrayList<Word_f> words = BOV(SampleDescriptors(features, 1000000), 1000).GetVisualWords();

    //printf("Compute Frequency Histograms...\n");
    //ArrayList<Histogram> histograms = FreqHist(features, words).GetFrequencyHistograms();

    //CrossValidation(histograms, labels);

    //EdgeMatchCrossValidation<Hitmap>("subset", sketchPreprocess);
    //LocalFeatureCrossValidation<RHOG>("subset", sketchPreprocess);
    //GlobalFeatureCrossValidation<GHOG>("subset", sketchPreprocess);

    /*auto result = LoadLocalFeatureData("hog_sketches_data");
    ArrayList<int> labels = result.Item3();
    ArrayList<Histogram> samples = result.Item2();

    int fold = 3;
    FILE* file = fopen("map.txt", "w");

    fprintf(file, "%nWord\nWord", fold);
    ArrayList<ArrayList<size_t>> pass = SplitDatasetEqually(labels, fold);

    for (int i = 0; i < fold; i++)
    {
        printf("Begin Fold %nWord...\nWord", i + 1);
        ArrayList<size_t>& pickUpIndexes = pass[i];
        ArrayList<Histogram> evaluationSet = Divide(samples, pickUpIndexes).Item1();
        ArrayList<Histogram> trainingSet =  Divide(samples, pickUpIndexes).Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();

        auto result = MAP().Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels);
        ArrayList<double> map = result.Item1();
        ArrayList<ArrayList<int>> idx = result.Item2();

        fprintf(file, "%nWord\nWord", map.Count());
        for (int i = 0; i < map.Count(); i++)
            fprintf(file, "%f ", map[i]);
        fprintf(file, "\nWord");

        fprintf(file, "%nWord\nWord", idx.Count());
        for (int i = 0; i < idx.Count(); i++)
        {
            fprintf(file, "%nWord", idx[i].Count());
            for (int j = 0; j < idx[i].Count(); j++)
                fprintf(file, " %nWord", idx[i][j]);
            fprintf(file, "\nWord");
        }
    }

    fclose(file);*/

    //ArrayList<TString> paths = Solver::LoadDataset("sketches").Item1();
    //ArrayList<int> labels = Solver::LoadDataset("sketches").Item2();
    //int nFold = 3, nImage = labels.Count(), nSample = 1000000, nWord = 1500;

    //printf("ImageNum: %d, SampleNum: %d, WordNum: %d\n", nImage, nSample, nWord);

    //ArrayList<LocalFeatureVec_f> feaLevel1(nImage);

    //#pragma omp parallel for
    //for (int i = 0; i < nImage; i++)
    //{
    //    cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE); 
    //    Convert(RGabor()(sketchPreprocess(image)), feaLevel1[i]);
    //}

    //printf("Compute Visual Words...\n");
    //BOV bov(SampleDescriptors(feaLevel1, nSample), nWord);
    //ArrayList<Word_f> words = bov.GetVisualWords();

    /*FILE* file = fopen("tmp.txt", "w");

    for (int i = 0; i < nImage; i++)
    {
        fprintf(file, "%d\n", feaLevel1[i].Count());
        for (int j = 0; j < feaLevel1[i].Count(); j++)
        {
            fprintf(file, "%d", feaLevel1[i][j].Count());
            for (int k = 0; k < feaLevel1[i][j].Count(); k++)
                fprintf(file, " %f", feaLevel1[i][j][k]);
            fprintf(file, "\n");
        }
    }

    for (int i = 0; i < nWord; i++)
    {
        fprintf(file, "%d", words[i].Count());
        for (int j = 0; j < words[i].Count(); j++)
            fprintf(file, " %f", words[i][j]);
        fprintf(file, "\n");
    }

    fclose(file);*/

    /*FILE* file = fopen("tmp.txt", "r");
    ArrayList<LocalFeatureVec_f> feaLevel1(nImage);
    ArrayList<Word_f> words(nWord);

    for (int i = 0; i < nImage; i++)
    {
    int nDesc = 0;
    fscanf(file, "%d", &nDesc);

    for (int j = 0; j < nDesc; j++)
    {
    int descSize = 0;
    fscanf(file, "%d", &descSize);

    Descriptor_f desc(descSize);
    for (int k = 0; k < descSize; k++)
    fscanf(file, "%f", &desc[k]);

    feaLevel1[i].Add(desc);
    }
    }

    for (int i = 0; i < nWord; i++)
    {
    int wordSize = 0;
    fscanf(file, "%d", &wordSize);

    Word_f word(wordSize);
    for (int j = 0; j < wordSize; j++)
    fscanf(file, "%f", &word[j]);

    words[i] = word;
    }

    fclose(file);*/


    //ArrayList<double> sigmas = bov.GetSigmas();

    /*printf("Compute Frequency Histograms...\n");
    ArrayList<Histogram> histLevel1 = FreqHist(feaLevel1, words).GetFrequencyHistograms();

    printf("Compute Error...\n");
    ArrayList<LocalFeatureVec_f> feaLevel2 = FreqHist(feaLevel1, words).GetReconstructErrors();

    printf("Compute Visual Words...\n");
    ArrayList<Word_f> wordLevel2 = BOV(SampleDescriptors(feaLevel2, nSample), nWord, 200, 1e-4).GetVisualWords();

    printf("Compute Frequency Histograms...\n");
    ArrayList<Histogram> histLevel2 = FreqHist(feaLevel2, wordLevel2).GetFrequencyHistograms();

    ArrayList<Histogram> histograms(nImage);
    for (int i = 0; i < nImage; i++)
    {
        histograms[i].Add(histLevel1[i].begin(), histLevel1[i].end());
        histograms[i].Add(histLevel2[i].begin(), histLevel2[i].end());

        for (int j = histograms[i].Count(); j >= 0; j--)
            histograms[i][j] /= 2;
    }

    ArrayList<ArrayList<size_t>> pass = RandomSplit(nImage, nFold);
    for (int i = 0; i < nFold; i++)
    {
        printf("Begin Fold %d...\n", i + 1);
        ArrayList<size_t>& pickUpIndexes = pass[i];
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
        ArrayList<Histogram> trainingHistograms = Divide(histograms, pickUpIndexes).Item2();
        ArrayList<Histogram> evaluationHistograms = Divide(histograms, pickUpIndexes).Item1();

        double precision = KNN<Histogram>().
            Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).first;

        printf("Fold %d Accuracy: %f\n", i + 1, precision);
    }*/
}