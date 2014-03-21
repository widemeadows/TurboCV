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
void EnLocalFeatureCrossValidation(const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    EnLocalFeatureSolver<FeatureType> solver(preprocess, datasetPath);
    solver.CrossValidation();

    TString savePath = FeatureType().GetName() + "_e_" + datasetPath;

    // Save KNN Accuracies

    ArrayList<double> accuracies = solver.GetAccuracies();
    FILE* file = fopen(savePath + "_knn.out", "w");

    for (int i = 0; i < accuracies.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, accuracies[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(accuracies), 
        Math::StandardDeviation(accuracies));

    fclose(file);

    // Save Mean Average Precision

    ArrayList<double> precisions = solver.GetPrecisions();
    file = fopen(savePath + "_map.out", "w");

    for (int i = 0; i < precisions.Count(); i++)
        fprintf(file, "%f ", precisions[i]);
    fprintf(file, "\n");

    fclose(file);

#if defined(SAVE_FEATURE)
    ArrayList<Word_f> words = solver.GetWords();
    ArrayList<Histogram> histograms = solver.GetHistograms();
    ArrayList<int> labels = solver.GetLabels();

    SaveLocalFeatures(savePath + "_data", words, histograms, labels);
#endif
}

template<typename FeatureType>
void LocalFeatureCrossValidation(const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    LocalFeatureSolver<FeatureType> solver(preprocess, datasetPath, "config.xml");
    solver.CrossValidation();

    TString savePath = FeatureType().GetName() + "_" + datasetPath;

    // Save KNN Accuracies

    ArrayList<double> accuracies = solver.GetAccuracies();
    FILE* file = fopen(savePath + "_knn.out", "w");

    for (int i = 0; i < accuracies.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, accuracies[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(accuracies), 
        Math::StandardDeviation(accuracies));

    fclose(file);

    //// Save Mean Average Precision

    //ArrayList<double> precisions = solver.GetPrecisions();
    //file = fopen(savePath + "_map.out", "w");

    //for (int i = 0; i < precisions.Count(); i++)
    //    fprintf(file, "%f ", precisions[i]);
    //fprintf(file, "\n");

    //fclose(file);

#if defined(SAVE_FEATURE) || defined(SAVE_DISTANCE_MATRIX)
    ArrayList<Word_f> words = solver.GetWords();
    ArrayList<Histogram> histograms = solver.GetHistograms();
    ArrayList<int> labels = solver.GetLabels();

#if defined(SAVE_FEATURE)
    SaveLocalFeatures(savePath + "_data", words, histograms, labels);
#endif

#if defined(SAVE_DISTANCE_MATRIX)
    SaveDistanceMatrix(savePath + "_matrix", histograms, labels);
#endif

#endif
}

template<typename FeatureType>
void GlobalFeatureCrossValidation(const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    GlobalFeatureSolver<FeatureType> solver(preprocess, datasetPath);
    solver.CrossValidation();

    TString savePath = FeatureType().GetName() + "_" + datasetPath;

    // Save KNN Accuracies

    ArrayList<double> accuracies = solver.GetAccuracies();
    FILE* file = fopen(savePath + "_knn.out", "w");

    for (int i = 0; i < accuracies.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, accuracies[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(accuracies), 
        Math::StandardDeviation(accuracies));

    fclose(file);

    // Save Mean Average Precision

    ArrayList<double> precisions = solver.GetPrecisions();
    file = fopen(savePath + "_map.out", "w");

    for (int i = 0; i < precisions.Count(); i++)
        fprintf(file, "%f ", precisions[i]);
    fprintf(file, "\n");

    fclose(file);

#if defined(SAVE_FEATURE) || defined(SAVE_DISTANCE_MATRIX)
    ArrayList<GlobalFeatureVec_f> features = solver.GetFeatures();
    ArrayList<int> labels = solver.GetLabels();

#if defined(SAVE_FEATURE)
    SaveGlobalFeatures(savePath + "_data", features, labels);
#endif

#if defined(SAVE_DISTANCE_MATRIX)
    SaveDistanceMatrix(savePath + "_matrix", features, labels);
#endif

#endif
}

template<typename EdgeMatchType>
void EdgeMatchCrossValidation(const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    EdgeMatchSolver<EdgeMatchType> solver(preprocess, datasetPath);
    solver.CrossValidation();

    TString savePath = EdgeMatchType().GetName() + "_" + datasetPath;

    // Save KNN Accuracies

    ArrayList<double> accuracies = solver.GetAccuracies();
    FILE* file = fopen(savePath + "_knn.out", "w");

    for (int i = 0; i < accuracies.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, accuracies[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(accuracies), 
        Math::StandardDeviation(accuracies));

    fclose(file);

    // Save Mean Average Precision

    ArrayList<double> precisions = solver.GetPrecisions();
    file = fopen(savePath + "_map.out", "w");

    for (int i = 0; i < precisions.Count(); i++)
        fprintf(file, "%f ", precisions[i]);
    fprintf(file, "\n");

    fclose(file);

#if defined(SAVE_DISTANCE_MATRIX)
    cv::Mat distanceMatrix = solver.GetDistanceMatrix();
    ArrayList<int> labels = solver.GetLabels();

    SaveDistanceMatrix(savePath + "_matrix", distanceMatrix, labels);
#endif
}

template<typename T>
void CrossValidation(const ArrayList<T>& samples, const ArrayList<int>& labels, int nFold = 3)
{
    ArrayList<ArrayList<size_t>> evaIdxes = RandomSplit(labels.Count(), nFold);
    ArrayList<double> precisions;

    for (int i = 0; i < nFold; i++)
    {
        printf("Begin Fold %d...\n", i + 1);
        const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
        ArrayList<T> trainingSet = Divide(samples, pickUpIndexes).Item2();
        ArrayList<T> evaluationSet = Divide(samples, pickUpIndexes).Item1();
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();

        precisions.Add(KNN<T>().
            Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels).Item1());

        printf("Fold %d Accuracy: %f\n\n", i + 1, precisions[i]);
    }

    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(precisions), 
        Math::StandardDeviation(precisions));
}