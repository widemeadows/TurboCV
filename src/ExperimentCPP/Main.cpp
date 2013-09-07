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

//Mat Preprocess(const Mat& sketchImage, bool thinning, Size size)
//{
//    assert(size.width == size.height);
//
//    Mat tmpImage = reverse(sketchImage);
//    resize(tmpImage, tmpImage, Size(256, 256));
//
//    //threshold(tmpImage, tmpImage, 200, 1, CV_THRESH_BINARY_INV);
//
//    //Mat binaryImage;
//    //threshold(sketchImage, binaryImage, 200, 1, CV_THRESH_BINARY_INV);
//
//    //Mat boundingBox = GetBoundingBox(binaryImage);
//
//    //Mat squareImage;
//    //int widthPadding = 0, heightPadding = 0;
//    //if (boundingBox.rows < boundingBox.cols)
//    //    heightPadding = (boundingBox.cols - boundingBox.rows) / 2;
//    //else
//    //    widthPadding = (boundingBox.rows - boundingBox.cols) / 2;
//    //copyMakeBorder(boundingBox, squareImage, heightPadding, heightPadding, 
//    //    widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
//
//    //Mat scaledImage;
//    //Size scaledSize = Size((int)(size.width - 2 * size.width / 18.0),
//    //    (int)(size.height - 2 * size.height / 18.0));
//    //resize(squareImage, scaledImage, scaledSize);
//
//    //Mat paddedImage;
//    //heightPadding = (size.height - scaledSize.height) / 2,
//    //    widthPadding = (size.width - scaledSize.width) / 2; 
//    //copyMakeBorder(scaledImage, paddedImage, heightPadding, heightPadding, 
//    //    widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
//    //assert(paddedImage.rows == size.height && paddedImage.cols == size.width);
//
//    Mat finalImage = tmpImage;
//    //clean(paddedImage, finalImage, 3);
//
//    if (thinning)
//        thin(finalImage, finalImage);
//
//    return finalImage;
//}

template<typename EdgeMatchType>
void EdgeMatchCrossValidation(cv::Mat (*Preprocess)(const cv::Mat&), const TString& datasetPath)
{
    EdgeMatchSolver<EdgeMatchType> solver(Preprocess, datasetPath);
    solver.CrossValidation();

    TString savePath = EdgeMatchType().GetName() + "_" + datasetPath + "_knn.out";

    ArrayList<double> precisions = solver.GetPrecisions();
    FILE* file = fopen(savePath, "w");

    for (int i = 0; i < precisions.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, precisions[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
        Math::StandardDeviation(precisions));

    fclose(file);

#if defined(SAVE_DISTANCE_MATRIX)
    cv::Mat distanceMatrix = solver.GetDistanceMatrix();
    ArrayList<int> labels = solver.GetLabels();

    savePath = EdgeMatchType().GetName() + "_" + datasetPath + "_matrix";
    SaveDistanceMatrix(savePath, distanceMatrix, labels);
#endif
}

void Batch()
{
    LocalFeatureCrossValidation<HOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RHOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<SHOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<LogSHOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<HOOSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RHOOSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<SC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<PSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RPSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RGabor>(sketchPreprocess, "sketches");
    printf("\n");

    GlobalFeatureCrossValidation<GHOG>(sketchPreprocess, "sketches");
    printf("\n");

    GlobalFeatureCrossValidation<GIST>(sketchPreprocess, "sketches");
    printf("\n");
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

int main()
{
    EdgeMatchCrossValidation<Hitmap>(sketchPreprocess, "sketches");
    //LocalFeatureCrossValidation<RHOG>(sketchPreprocess, "sketches");
    //GlobalFeatureCrossValidation<GHOG>(sketchPreprocess, "subset");

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