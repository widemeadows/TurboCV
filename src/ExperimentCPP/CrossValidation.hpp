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
void LocalFeatureCrossValidation(cv::Mat (*Preprocess)(const cv::Mat&), const TString& datasetPath)
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
void GlobalFeatureCrossValidation(cv::Mat (*Preprocess)(const cv::Mat&), const TString& datasetPath)
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

template<typename T>
void CrossValidation(const ArrayList<T>& samples, const ArrayList<int>& labels, int fold = 3)
{
    ArrayList<Group<ArrayList<T>, ArrayList<T>, ArrayList<size_t>>> pass = 
        RandomSplit(samples, fold);
    ArrayList<double> passResult;

    for (int i = 0; i < fold; i++)
    {
        printf("Begin Fold %nDesc...\n", i + 1);
        ArrayList<T>& evaluationSet = pass[i].Item1();
        ArrayList<T>& trainingSet = pass[i].Item2();
        ArrayList<size_t>& pickUpIndexes = pass[i].Item3();

        ArrayList<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int k = 0; k < samples.Count(); k++)
        {
            if (counter < pickUpIndexes.Count() && k == pickUpIndexes[counter])
            {
                evaluationLabels.Add(labels[k]);
                counter++;
            }
            else
                trainingLabels.Add(labels[k]);
        }

        KNN<T> knn;
        auto precisions = knn.Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels);

        passResult.Add(precisions.first);
        printf("Fold %nDesc Accuracy: %f\n\n", i + 1, precisions.first);
    }

    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
}