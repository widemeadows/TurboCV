#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include <cv.h>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace cv;

#define SAVE_FEATURE
//#define SAVE_DISTANCE_MATRIX

template<typename FeatureType>
void LocalFeatureCrossValidation(const TString& datasetPath, cv::Mat (*Preprocess)(const cv::Mat&))
{
    LocalFeatureSolver<FeatureType> solver(Preprocess, datasetPath);
    solver.CrossValidation();

    TString savePath = FeatureType().GetName() + "_" + datasetPath + "_knn.out";

    ArrayList<double> precisions = solver.GetPrecisions();
    FILE* file = fopen(savePath, "w");

    for (int i = 0; i < precisions.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, precisions[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
        Math::StandardDeviation(precisions));

    fclose(file);

#if defined(SAVE_FEATURE) || defined(SAVE_DISTANCE_MATRIX)
    ArrayList<Word_f> words = solver.GetWords();
    ArrayList<Histogram> histograms = solver.GetHistograms();
    ArrayList<int> labels = solver.GetLabels();

#if defined(SAVE_FEATURE)
    savePath = FeatureType().GetName() + "_" + datasetPath + "_data";
    SaveLocalFeatures(savePath, words, histograms, labels);
#endif

#if defined(SAVE_DISTANCE_MATRIX)
    savePath = FeatureType().GetName() + "_" + datasetPath + "_matrix";
    SaveDistanceMatrix(savePath, histograms, labels);
#endif

#endif
}

template<typename FeatureType>
void GlobalFeatureCrossValidation(const TString& datasetPath, cv::Mat (*Preprocess)(const cv::Mat&))
{
    GlobalFeatureSolver<FeatureType> solver(Preprocess, datasetPath);
    solver.CrossValidation();

    TString savePath = FeatureType().GetName() + "_" + datasetPath + "_knn.out";

    ArrayList<double> precisions = solver.GetPrecisions();
    FILE* file = fopen(savePath, "w");

    for (int i = 0; i < precisions.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, precisions[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
        Math::StandardDeviation(precisions));

    fclose(file);

#if defined(SAVE_FEATURE) || defined(SAVE_DISTANCE_MATRIX)
    ArrayList<GlobalFeatureVec_f> features = solver.GetFeatures();
    ArrayList<int> labels = solver.GetLabels();

#if defined(SAVE_FEATURE)
    savePath = FeatureType().GetName() + "_" + datasetPath + "_data";
    SaveGlobalFeatures(savePath, features, labels);
#endif

#if defined(SAVE_DISTANCE_MATRIX)
    savePath = FeatureType().GetName() + "_" + datasetPath + "_matrix";
    SaveDistanceMatrix(savePath, features, labels);
#endif

#endif
}

template<typename EdgeMatchType>
void EdgeMatchCrossValidation(const TString& datasetPath, cv::Mat (*Preprocess)(const cv::Mat&))
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

template<typename T>
void CrossValidation(const ArrayList<T>& samples, const ArrayList<int>& labels, int nFold = 3)
{
    ArrayList<ArrayList<size_t>> evaIdxes = RandomSplit(labels.Count(), nFold);
    ArrayList<double> precisions;

    for (int i = 0; i < nFold; i++)
    {
        printf("Begin Fold %nDesc...\n", i + 1);
        const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
        ArrayList<T> trainingSet = Divide(samples, pickUpIndexes).Item2();
        ArrayList<T> evaluationSet = Divide(samples, pickUpIndexes).Item1();
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();

        precisions.Add(KNN<T>().
            Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels).Item1());

        printf("Fold %nDesc Accuracy: %f\n\n", i + 1, precisions[i]);
    }

    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
        Math::StandardDeviation(precisions));
}