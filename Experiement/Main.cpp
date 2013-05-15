#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include "Util.h"
#include "TSNE.h"
#include <cv.h>
#include <highgui.h>
using namespace TurboCV::System;
using namespace TurboCV::System::IO;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;
using namespace cv;

inline TurboCV::System::ArrayList<Tuple<Mat, int>> GetImages(
    const TurboCV::System::String& imageSetPath, 
    int imageLoadMode = CV_LOAD_IMAGE_GRAYSCALE)
{
    DirectoryInfo imageSetInfo(imageSetPath);

    ArrayList<TurboCV::System::String> classInfos = imageSetInfo.GetDirectories();
    sort(classInfos.begin(), classInfos.end());

    ArrayList<Tuple<Mat, int>> images;
    for (int i = 0; i < classInfos.Count(); i++)
    {
        ArrayList<TurboCV::System::String> fileInfos = DirectoryInfo(classInfos[i]).GetFiles();
        sort(fileInfos.begin(), fileInfos.end());

        for (int j = 0; j < fileInfos.Count(); j++)
            images.Add(CreateTuple(imread(fileInfos[j], imageLoadMode), i + 1));
    }

    return images;
}

void NormlizeDeviation(ArrayList<Histogram> vecs)
{
    Histogram mean(vecs[0].Count());
    double deviation = 0;

    for (size_t i = 0; i < vecs.Count(); i++)
        for (size_t j = 0; j < vecs[i].Count(); j++)
            mean[j] += vecs[i][j];
    
    for (size_t i = 0; i < mean.Count(); i++)
        mean[i] /= vecs.Count();

    for (size_t i = 0; i < vecs.Count(); i++)
        for (size_t j = 0; j < vecs[i].Count(); j++)
            deviation += (vecs[i][j] - mean[j]) * (vecs[i][j] - mean[j]);

    deviation = sqrt(deviation / vecs.Count());

    for (size_t i = 0; i < vecs.Count(); i++)
        for (size_t j = 0; j < vecs[i].Count(); j++)
            vecs[i][j] /= deviation;
}

template<typename LocalFeature>
void LocalFeatureCrossValidation(const TurboCV::System::String& imageSetPath, const LocalFeature& algo, 
                                 int wordNum, bool thinning = false, int sampleNum = 1000000, int fold = 3)
{
    srand(1);
    ArrayList<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.Count();

    TurboCV::System::ArrayList<LocalFeature_f> features(imageNum);
    printf("Compute " + algo.GetName() + "...\n");
    LocalFeature machine = algo;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
    {
        Convert(machine.GetFeatureWithPreprocess(images[i].Item1(), thinning), features[i]);
        images[i].Item1().release();
    }

#ifdef SAVE_FEATURE
    { // Use a block here to destruct words and freqHistograms immediately.
        printf("Compute Visual Words...\n");
        ArrayList<Word_f> words = BOV::GetVisualWords(features, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        ArrayList<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

        printf("Write To File...\n");
        TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath;
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < freqHistograms.Count(); i++)
        {
            fprintf(file, "%d", images[i].Item2());
            for (int j = 0; j < freqHistograms[i].Count(); j++)
                fprintf(file, " %d:%f", j + 1, freqHistograms[i][j]);
            fprintf(file, "\n");
        }
        fclose(file);
    }
#endif

#ifdef SAVE_DISTANCE_MATRIX
    {
        ArrayList<Word_f> words = BOV::GetVisualWords(features, wordNum, sampleNum);

        ArrayList<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

        Mat distanceMatrix = Mat::zeros(imageNum, imageNum, CV_64F);
        #pragma omp parallel for
        for (int i = 0; i < imageNum; i++)
        {
            for (int j = 0; j < imageNum; j++)
            {
                if (i != j)
                    distanceMatrix.at<double>(i, j) = 
                        Math::NormOneDistance(freqHistograms[i], freqHistograms[j]);
            }
        }

        printf("Write To File...\n");
        TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath + "_matrix";
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < imageNum; i++)
        {
            fprintf(file, "%d", images[i].Item2());
            for (int j = 0; j < imageNum; j++)
            {
                if (i != j)
                {
                    fprintf(file, " %d:%f", images[i].Item2() == images[j].Item2() ? 1 : 0, 
                        distanceMatrix.at<double>(i, j));
                }
            }
            fprintf(file, "\n");
        }
        fclose(file);
        distanceMatrix.release();
    }
#endif

    ArrayList<Tuple<ArrayList<LocalFeature_f>, ArrayList<LocalFeature_f>, ArrayList<size_t>>> pass = 
        RandomSplit(features, fold);

#ifdef SAVE_ROC
    ArrayList<ArrayList<double>> DRs(imageNum), FPRs(imageNum);
#endif

    ArrayList<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        ArrayList<LocalFeature_f>& evaluationSet = pass[i].Item1();
        ArrayList<LocalFeature_f>& trainingSet = pass[i].Item2();
        ArrayList<size_t>& pickUpIndexes = pass[i].Item3();

        printf("Compute Visual Words...\n");
        ArrayList<Word_f> words = BOV::GetVisualWords(trainingSet, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        ArrayList<Histogram> trainingHistograms = BOV::GetFrequencyHistograms(trainingSet, words);
        ArrayList<Histogram> evaluationHistograms = BOV::GetFrequencyHistograms(evaluationSet, words);
        
        ArrayList<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.Count() && j == pickUpIndexes[counter])
            {
                evaluationLabels.Add(images[j].Item2());
                counter++;
            }
            else
                trainingLabels.Add(images[j].Item2());
        }

        KNN<Histogram> knn;
        pair<double, map<pair<int, int>, double>> precisions = 
            knn.Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels);
       
        passResult.Add(precisions.first);
        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);

#ifdef SAVE_ROC
        pair<ArrayList<ArrayList<double>>, ArrayList<ArrayList<bool>>> matrix =
            GetDistanceMatrix(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels);
        ArrayList<ArrayList<double>>& distances = matrix.first;
        ArrayList<ArrayList<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.Count(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = roc.Item1();
            FPRs[index] = roc.Item2();
        }
#endif
    }

#ifdef SAVE_ROC
    {
        TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath + "_roc";
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < DRs.Count(); i++)
        {
            for (int j = 0; j < DRs[i].Count(); j++)
                fprintf(file, "%f ", DRs[i][j]);
            fprintf(file, "\n");

            for (int j = 0; j < FPRs[i].Count(); j++)
                fprintf(file, "%f ", FPRs[i][j]);
            fprintf(file, "\n");
        }
        fclose(file);
    }
#endif

    TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath + "_knn.out";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < passResult.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    fclose(file);
}

template<typename GlobalFeature>
void GlobalFeatureCrossValidation(const TurboCV::System::String& imageSetPath, const GlobalFeature& algo, 
                                  bool thinning = false, int fold = 3)
{
    srand(1);
    ArrayList<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.Count();

    ArrayList<GlobalFeature_f> features(imageNum);
    printf("Compute " + algo.GetName() + "...\n");
    GlobalFeature machine = algo;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
    {
        Convert(machine.GetFeatureWithPreprocess(images[i].Item1(), thinning, Size(64, 64)), features[i]);
        images[i].Item1().release();
    }

#ifdef SAVE_FEATURE
    {
        printf("Write To File...\n");
        TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath;
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < features.Count(); i++)
        {
            fprintf(file, "%d", images[i].Item2());
            for (int j = 0; j < features[i].Count(); j++)
                fprintf(file, " %d:%f", j + 1, features[i][j]);
            fprintf(file, "\n");
        }
        fclose(file);
    }
#endif

#ifdef SAVE_DISTANCE_MATRIX
    {
        Mat distanceMatrix = Mat::zeros(imageNum, imageNum, CV_64F);
        #pragma omp parallel for
        for (int i = 0; i < imageNum; i++)
        {
            for (int j = 0; j < imageNum; j++)
            {
                if (i != j)
                    distanceMatrix.at<double>(i, j) = 
                        Math::NormOneDistance(features[i], features[j]);
            }
        }

        printf("Write To File...\n");
        TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath + "_matrix";
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < imageNum; i++)
        {
            fprintf(file, "%d", images[i].Item2());
            for (int j = 0; j < imageNum; j++)
            {
                if (i != j)
                {
                    fprintf(file, " %d:%f", images[i].Item2() == images[j].Item2() ? 1 : 0, 
                        distanceMatrix.at<double>(i, j));
                }
            }
            fprintf(file, "\n");
        }
        fclose(file);
        distanceMatrix.release();
    }
#endif

    ArrayList<Tuple<ArrayList<GlobalFeature_f>, ArrayList<GlobalFeature_f>, ArrayList<size_t>>> pass = 
        RandomSplit(features, fold);

#ifdef SAVE_ROC
    ArrayList<ArrayList<double>> DRs(imageNum), FPRs(imageNum);
#endif

    ArrayList<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        ArrayList<GlobalFeature_f>& evaluationSet = pass[i].Item1();
        ArrayList<GlobalFeature_f>& trainingSet = pass[i].Item2();
        ArrayList<size_t>& pickUpIndexes = pass[i].Item3();

        ArrayList<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.Count() && j == pickUpIndexes[counter])
            {
                evaluationLabels.Add(images[j].Item2());
                counter++;
            }
            else
                trainingLabels.Add(images[j].Item2());
        }

        ArrayList<size_t> actualIndexes = RandomPermutate(evaluationSet.Count(), 10000);
        ArrayList<GlobalFeature_f> actualEvaSet;
        ArrayList<int> actualEvaLabels;
        for (int i = 0; i < actualIndexes.Count(); i++)
        {
            actualEvaSet.Add(evaluationSet[actualIndexes[i]]);
            actualEvaLabels.Add(evaluationLabels[actualIndexes[i]]);
        }

        MQDF<GlobalFeature_f> mqdf;
        pair<double, map<pair<int, int>, double>> precisions = 
            mqdf.Evaluate(trainingSet, trainingLabels, actualEvaSet, actualEvaLabels);
        //KNN<GlobalFeature_f> knn;
        //pair<double, map<pair<int, int>, double>> precisions = 
        //    knn.Evaluate(trainingSet, trainingLabels, actualEvaSet, actualEvaLabels,
        //    Math::NormOneDistance, 4.0, 4);

        passResult.Add(precisions.first);
        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);

#ifdef SAVE_ROC
        pair<ArrayList<ArrayList<double>>, ArrayList<ArrayList<bool>>> matrix =
            GetDistanceMatrix(trainingSet, trainingLabels, evaluationSet, evaluationLabels);
        ArrayList<ArrayList<double>>& distances = matrix.first;
        ArrayList<ArrayList<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.Count(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = roc.Item1();
            FPRs[index] = roc.Item2();
        }
#endif
    }

#ifdef SAVE_ROC
    {
        TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath + "_roc";
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < DRs.Count(); i++)
        {
            for (int j = 0; j < DRs[i].Count(); j++)
                fprintf(file, "%f ", DRs[i][j]);
            fprintf(file, "\n");

            for (int j = 0; j < FPRs[i].Count(); j++)
                fprintf(file, "%f ", FPRs[i][j]);
            fprintf(file, "\n");
        }
        fclose(file);
    }
#endif

    TurboCV::System::String savePath = algo.GetName() + "_" + imageSetPath + "_knn.out";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < passResult.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    fclose(file);
}

template<typename EdgeMatching>
void EdgeMatchingCrossValidation(const TurboCV::System::String& imageSetPath, const EdgeMatching& matching,
                                 bool thinning = false, int fold = 3)
{
    srand(1);
    ArrayList<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.Count();

    ArrayList<EdgeMatching::Info> transforms(imageNum);
    printf("Compute " + matching.GetName() + "...\n");
    EdgeMatching machine = matching;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
    {
        transforms[i] = machine.GetFeatureWithPreprocess(images[i].Item1(), thinning);
        images[i].Item1().release();
    }

#ifdef SAVE_DISTANCE_MATRIX
    {
        Mat distanceMatrix = Mat::zeros(imageNum, imageNum, CV_64F);
        #pragma omp parallel for
        for (int i = 0; i < imageNum; i++)
        {
            for (int j = 0; j < imageNum; j++)
            {
                if (i != j)
                    distanceMatrix.at<double>(i, j) = EdgeMatching::GetDistance(transforms[i], transforms[j]);
            }
        }

        printf("Write To File...\n");
        TurboCV::System::String savePath = matching.GetName() + "_" + imageSetPath + "_matrix";
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < imageNum; i++)
        {
            fprintf(file, "%d", images[i].Item2());
            for (int j = 0; j < imageNum; j++)
            {
                if (i != j)
                {
                    fprintf(file, " %d:%f", images[i].Item2() == images[j].Item2() ? 1 : 0, 
                        distanceMatrix.at<double>(i, j));
                }
            }
            fprintf(file, "\n");
        }
        fclose(file);
        distanceMatrix.release();
    }
#endif

    ArrayList<Tuple<ArrayList<EdgeMatching::Info>, ArrayList<EdgeMatching::Info>, ArrayList<size_t>>> pass = 
        RandomSplit(transforms, fold);

#ifdef SAVE_ROC
    ArrayList<ArrayList<double>> DRs(imageNum), FPRs(imageNum);
#endif

    ArrayList<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        ArrayList<EdgeMatching::Info>& evaluationSet = pass[i].Item1();
        ArrayList<EdgeMatching::Info>& trainingSet = pass[i].Item2();
        ArrayList<size_t>& pickUpIndexes = pass[i].Item3();

        ArrayList<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.Count() && j == pickUpIndexes[counter])
            {
                evaluationLabels.Add(images[j].Item2());
                counter++;
            }
            else
                trainingLabels.Add(images[j].Item2());
        }

        KNN<EdgeMatching::Info> knn;
        pair<double, map<pair<int, int>, double>> precisions = knn.Evaluate(
            trainingSet, trainingLabels, evaluationSet, evaluationLabels, 
            EdgeMatching::GetDistance, KNN<EdgeMatching::Info>::HARD_VOTING);

        passResult.Add(precisions.first);
        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);

#ifdef SAVE_ROC
        pair<ArrayList<ArrayList<double>>, ArrayList<ArrayList<bool>>> matrix = GetDistanceMatrix(
            trainingSet, trainingLabels, evaluationSet, evaluationLabels, EdgeMatching::GetDistance);
        ArrayList<ArrayList<double>>& distances = matrix.first;
        ArrayList<ArrayList<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.Count(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = roc.Item1();
            FPRs[index] = roc.Item2();
        }
#endif
    }

#ifdef SAVE_ROC
    {
        TurboCV::System::String savePath = matching.GetName() + "_" + imageSetPath + "_roc";
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < DRs.Count(); i++)
        {
            for (int j = 0; j < DRs[i].Count(); j++)
                fprintf(file, "%f ", DRs[i][j]);
            fprintf(file, "\n");

            for (int j = 0; j < FPRs[i].Count(); j++)
                fprintf(file, "%f ", FPRs[i][j]);
            fprintf(file, "\n");
        }
        fclose(file);
    }
#endif

    TurboCV::System::String savePath = matching.GetName() + "_" + imageSetPath + "_knn.out";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < passResult.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    fclose(file);
}

//ArrayList<double> Boosting(const ArrayList<Histogram>& data, const ArrayList<int>& labels)
//{
//    ArrayList<Tuple<ArrayList<Histogram>, ArrayList<Histogram>, ArrayList<size_t>>> pass = 
//        RandomSplit(data, 3);
//
//    int histSize = data[0].size();
//    ArrayList<double> weights(histSize);
//
//    for (int k = 0; k < histSize; k++)
//    {
//        ArrayList<Histogram>& evaluationSet = pass[0].Item1();
//        ArrayList<Histogram>& trainingSet = pass[0].Item2();
//        ArrayList<size_t>& pickUpIndexes = pass[0].Item3();
//
//        ArrayList<int> trainingLabels, evaluationLabels;
//        int counter = 0;
//        for (int j = 0; j < data.size(); j++)
//        {
//            if (counter < pickUpIndexes.size() && j == pickUpIndexes[counter])
//            {
//                evaluationLabels.push_back(labels[j]);
//                counter++;
//            }
//            else
//                trainingLabels.push_back(labels[j]);
//        }
//
//        ArrayList<Histogram> evaluationData(evaluationSet.size()), trainingData(trainingSet.size());
//        for (int j = 0; j < evaluationSet.size(); j++)
//            evaluationData[j].push_back(evaluationSet[j][k]);
//        for (int j = 0; j < trainingSet.size(); j++)
//            trainingData[j].push_back(trainingSet[j][k]);
//
//        KNN<Histogram> knn;
//        pair<double, map<int, double>> precisions = 
//            knn.Evaluate(4, trainingData, trainingLabels, evaluationData, evaluationLabels);
//
//        weights[k] = precisions.first;
//    }
//
//    return weights;
//}


ArrayList<ArrayList<LocalFeature_f>> DivideLocalFeatures(const ArrayList<LocalFeature_f> features)
{
    int blockSize = 128;

    ArrayList<Point> wordCenters;
    for (int i = 1; i <= 3; i++)
        for (int j = 1; j <= 3; j++)
            wordCenters.Add(Point(i * 64, j * 64));

    ArrayList<Point> descPoints = SampleOnGrid(256, 256, 28);
    assert(descPoints.Count() == features[0].Count());

    ArrayList<ArrayList<LocalFeature_f>> parts(wordCenters.Count());
    for (int i = 0; i < features.Count(); i++)
    {
        ArrayList<LocalFeature_f> tmp(wordCenters.Count());

        for (int j = 0; j < wordCenters.Count(); j++)
        {
            int top = wordCenters[j].y - blockSize / 2,
                bottom = wordCenters[j].y + blockSize / 2,
                left = wordCenters[j].x - blockSize / 2,
                right = wordCenters[j].x + blockSize / 2;

            for (int k = 0; k < descPoints.Count(); k++)
            {
                if (top <= descPoints[k].y && descPoints[k].y < bottom &&
                    left <= descPoints[k].x && descPoints[k].x < right)
                    tmp[j].Add(features[i][k]);
            }
        }

        for (int j = 0; j < wordCenters.Count(); j++)
            parts[j].Add(tmp[j]);
    }

    return parts;
}

template<typename LocalFeature>
void LocalFeatureTest(const TurboCV::System::String& imageSetPath, const LocalFeature& feature, 
                        int wordNum, int sampleNum = 1000000, int fold = 3)
{
    srand(1);
    ArrayList<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.Count();

    ArrayList<LocalFeature_f> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    LocalFeature machine = feature;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
        Convert(machine.GetFeatureWithPreprocess(images[i].Item1(), true), features[i]);

    { // Use a block here to destruct words and freqHistograms immediately.
        printf("Compute Visual Words...\n");
        ArrayList<Word_f> words = BOV::GetVisualWords(features, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        ArrayList<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

        printf("Write To File...\n");
        TurboCV::System::String savePath = feature.GetName() + "_features";
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < freqHistograms.Count(); i++)
        {
            for (int j = 0; j < freqHistograms[i].Count(); j++)
                fprintf(file, "%.12f ", freqHistograms[i][j]);
            fprintf(file, "\n");
        }
        fclose(file);

        savePath = feature.GetName() + "_labels";
        file = fopen(savePath, "w");
        for (int i = 0; i < features.Count(); i++)
            fprintf(file, "%d\n", images[i].Item2());
        fclose(file);
    }

    ArrayList<Tuple<ArrayList<LocalFeature_f>, ArrayList<LocalFeature_f>, ArrayList<size_t>>> pass = 
        RandomSplit(features, fold);
    ArrayList<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        ArrayList<LocalFeature_f>& evaluationSet = pass[i].Item1();
        ArrayList<LocalFeature_f>& trainingSet = pass[i].Item2();
        ArrayList<size_t>& pickUpIndexes = pass[i].Item3();

        /*ArrayList<LocalFeature_f> evaluationSet1(evaluationSet.size());
        ArrayList<LocalFeature_f> evaluationSet2(evaluationSet.size());
        for (int i = 0; i < evaluationSet.size(); i++)
        {
            evaluationSet1[i].push_back(evaluationSet[i].begin(), 
                evaluationSet[i].begin() + evaluationSet[i].size() / 2);
            evaluationSet2[i].push_back(evaluationSet[i].begin() + evaluationSet[i].size() / 2, 
                evaluationSet[i].end());
        }

        ArrayList<LocalFeature_f> trainingSet1(trainingSet.size());
        ArrayList<LocalFeature_f> trainingSet2(trainingSet.size());
        for (int i = 0; i < trainingSet.size(); i++)
        {
            trainingSet1[i].push_back(trainingSet[i].begin(), 
                trainingSet[i].begin() + trainingSet[i].size() / 2);
            trainingSet2[i].push_back(trainingSet[i].begin() + trainingSet[i].size() / 2, 
                trainingSet[i].end());
        }*/

        printf("Compute Visual Words...\n");
        ArrayList<Word_f> words = BOV::GetVisualWords(trainingSet, wordNum, sampleNum);

        //ArrayList<Word_f> words1 = BOV::GetVisualWords(trainingSet1, wordNum, sampleNum);
        //ArrayList<Word_f> words2 = BOV::GetVisualWords(trainingSet2, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        ArrayList<Histogram> trainingHistograms = BOV::GetFrequencyHistograms(trainingSet, words)/* * 9*/;
        ArrayList<Histogram> evaluationHistograms = BOV::GetFrequencyHistograms(evaluationSet, words)/* * 9*/;

        //ArrayList<ArrayList<LocalFeature_f>> parts = DivideLocalFeatures(trainingSet);
        //for (int j = 0; j < parts.size(); j++)
        //{
        //    ArrayList<Histogram> result = BOV::GetFrequencyHistograms(parts[j], words);

        //    assert(result.size() == trainingHistograms.size());
        //    for (int k = 0; k < result.size(); k++)
        //        trainingHistograms[k].push_back(result[k]);
        //}

        //parts = DivideLocalFeatures(evaluationSet);
        //for (int j = 0; j < parts.size(); j++)
        //{
        //    ArrayList<Histogram> result = BOV::GetFrequencyHistograms(parts[j], words);

        //    assert(result.size() == evaluationHistograms.size());
        //    for (int k = 0; k < result.size(); k++)
        //        evaluationHistograms[k].push_back(result[k]);
        //}

        assert(trainingHistograms[0].Count() == wordNum &&
            evaluationHistograms[0].Count() == wordNum);

        /*ArrayList<Histogram> trainingHistograms1 = BOV::GetFrequencyHistograms(trainingSet1, words1);
        ArrayList<Histogram> trainingHistograms2 = BOV::GetFrequencyHistograms(trainingSet2, words2);

        ArrayList<Histogram> evaluationHistograms1 = BOV::GetFrequencyHistograms(evaluationSet1, words1);
        ArrayList<Histogram> evaluationHistograms2 = BOV::GetFrequencyHistograms(evaluationSet2, words2);

        ArrayList<Histogram> trainingHistograms(trainingSet.size()), evaluationHistograms(evaluationSet.size());

        NormlizeDeviation(trainingHistograms1);
        NormlizeDeviation(trainingHistograms2);
        for (int i = 0; i < trainingSet.size(); i++)
        {
            trainingHistograms[i].push_back(trainingHistograms1[i].begin(), trainingHistograms1[i].end());
            trainingHistograms[i].push_back(trainingHistograms2[i].begin(), trainingHistograms2[i].end());
        }

        NormlizeDeviation(evaluationHistograms1);
        NormlizeDeviation(evaluationHistograms2);
        for (int i = 0; i < evaluationSet.size(); i++)
        {
            evaluationHistograms[i].push_back(evaluationHistograms1[i].begin(), evaluationHistograms1[i].end());
            evaluationHistograms[i].push_back(evaluationHistograms2[i].begin(), evaluationHistograms2[i].end());
        }*/

        ArrayList<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int k = 0; k < imageNum; k++)
        {
            if (counter < pickUpIndexes.Count() && k == pickUpIndexes[counter])
            {
                evaluationLabels.Add(images[k].Item2());
                counter++;
            }
            else
                trainingLabels.Add(images[k].Item2());
        }

        //ArrayList<double> weights = Boosting(trainingHistograms, trainingLabels);
        //for (int k = 0; k < weights.size(); k++)
        //{
        //    for (int i = 0; i < trainingHistograms.size(); i++)
        //        trainingHistograms[i][k] *= weights[k];

        //    for (int i = 0; i < evaluationHistograms.size(); i++)
        //        evaluationHistograms[i][k] *= weights[k];
        //}

        //printf("Perform LDA...\n");
        //pair<ArrayList<Histogram>, ArrayList<Histogram>> result = LDAOperator::ComputeLDA(
        //    trainingHistograms, trainingLabels, 1000, evaluationHistograms);
        //trainingHistograms = result.first;
        //evaluationHistograms = result.second;

        KNN<Histogram> knn;
        auto precisions = 
            knn.Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels);

        passResult.Add(precisions.first);

        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);
    }

    TurboCV::System::String savePath = feature.GetName() + "_oracles_knn.out";
    FILE* file = fopen(savePath, "w");

    for (int i = 0; i < passResult.Count(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    fclose(file);
}

template<typename T>
void CrossValidation(const ArrayList<T>& samples, const ArrayList<int>& labels, int fold = 3)
{
	ArrayList<Tuple<ArrayList<T>, ArrayList<T>, ArrayList<size_t>>> pass = 
		RandomSplit(samples, fold);
    ArrayList<double> passResult;
    
	for (int i = 0; i < fold; i++)
    {
        printf("Begin Fold %d...\n", i + 1);
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
        auto precisions = knn.Evaluate(trainingSet, trainingLabels, evaluationSet, 
			evaluationLabels, Math::NormOneDistance, KNN<T>::HARD_VOTING);

        passResult.Add(precisions.first);
        printf("Fold %d Accuracy: %f\n\n", i + 1, precisions.first);
    }

	printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
}

void Batch(const TurboCV::System::String& imageSetPath, bool thinning = false)
{
    LocalFeatureCrossValidation(imageSetPath, HOG(), 500, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, RHOG(), 500, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, SHOG(), 500, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, LogSHOG(), 500, thinning);
    printf("\n");

    GlobalFeatureCrossValidation(imageSetPath, GHOG(), thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, HOOSC(), 1000, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, RHOOSC(), 1000, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, SC(), 1000, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, PSC(), 1000, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, RSC(), 1000, thinning);
    printf("\n");

    LocalFeatureCrossValidation(imageSetPath, PRSC(), 1000, thinning);
    printf("\n");

    GlobalFeatureCrossValidation(imageSetPath, GIST(), thinning);
    printf("\n");

    EdgeMatchingCrossValidation(imageSetPath, CM(), thinning);
    printf("\n");

    EdgeMatchingCrossValidation(imageSetPath, OCM(), thinning);
    printf("\n");

    EdgeMatchingCrossValidation(imageSetPath, Hitmap(), thinning);
    printf("\n");
}

int main()
{
	ArrayList<ArrayList<double>> samples;
	ArrayList<int> labels;
	double token;

	FILE* file = fopen("Y.txt", "r");
	while (fscanf(file, "%lf", &token) != EOF)
	{
		ArrayList<double> sample;

		sample.Add(token);
		for (int i = 1; i < 2; i++)
		{
			fscanf(file, "%lf", &token);
			sample.Add(token);
		}

		samples.Add(sample);
	}
	fclose(file);

	file = fopen("labels.txt", "r");
	while (fscanf(file, "%lf", &token) != EOF)
	{
		labels.Add(token);
	}
	fclose(file);

	assert(samples.Count() == labels.Count());
	CrossValidation(samples, labels);

	//file = fopen("Y.txt", "w");
	//TSNE tsne;
	//cv::Mat Y = tsne.Compute(samples);

	//for (int i = 0; i < Y.rows; i++)
	//{
	//	for (int j = 0; j < Y.cols; j++)
	//		fprintf(file, "%f ", Y.at<double>(i, j));
	//	fprintf(file, "\n");
	//}

	//fclose(file);

	//LocalFeatureCrossValidation("oracles", Test(), 1500, true);
	//printf("\n");

    //GlobalFeatureCrossValidation("hccr", GHOG(), true);
    //printf("\n");

    //EdgeMatchingCrossValidation("oracles", Hitmap(), true);
    //printf("\n");

	//LocalFeatureTest("oracles", Test(), 1500);
	//printf("\n");

    //Batch("sketches", false);
    //Batch("oracles", true);

    //Mat trans = getRotationMatrix2D(Point(image.rows / 2, image.cols / 2), -20, 1);
    //warpAffine(image, image, trans, image.size());
    //threshold(image, image, 0.1, 1, CV_THRESH_BINARY);
    //thin(image, image);
    //threshold(image, image, 0.1, 255, CV_THRESH_BINARY);

    /*ArrayList<Point> points = GetEdgels(image);
    ArrayList<int> xCount(image.cols);

    for (auto point : points)
        xCount[point.x]++;

    for (int i = 1; i < xCount.size(); i++)
        xCount[i] += xCount[i - 1];

    int blockNum = 4;
    double xStep = xCount.back() / (double)blockNum;

    ArrayList<int> xSep(blockNum + 1);
    int tmp = 1;
    for (int i = 0; i < xCount.size(); i++)
    {
        if (xCount[i] >= tmp * xStep)
        {
            xSep[tmp++] = i;

            if (tmp == blockNum)
                break;
        }
    }
    xSep[blockNum] = image.cols - 1;

    Mat ySep(blockNum + 1, blockNum, CV_32S);
    for (int i = 1; i < xSep.size(); i++)
    {
        int prevSep = xSep[i - 1], curSep = xSep[i];

        ArrayList<int> yCount(image.rows);
        for (auto point : points)
        {
            if (prevSep <= point.x && point.x < curSep)
                yCount[point.y]++;
        }

        for (int j = 1; j < yCount.size(); j++)
            yCount[j] += yCount[j - 1];

        double yStep = yCount.back() / (double)blockNum;
        int tmp = 1;
        for (int j = 0; j < yCount.size(); j++)
        {
            if (yCount[j] >= tmp * yStep)
            {
                ySep.at<int>(tmp++, i - 1) = j;

                if (tmp == blockNum)
                    break;
            }
        }
    }

    cvtColor(image, image, CV_GRAY2BGR);

    for (int i = 1; i < xSep.size() - 1; i++)
        line(image, Point(xSep[i], 0), Point(xSep[i], image.rows - 1), Scalar(0, 255, 0));

    for (int i = 1; i < ySep.rows - 1; i++)
        for (int j = 0; j < ySep.cols; j++)
            line(image, Point(xSep[j], ySep.at<int>(i, j)), Point(xSep[j + 1], ySep.at<int>(i, j)), Scalar(0, 255, 0));*/

    //for (auto point : points)
    //{
    //    circle(tmp, point, 1, Scalar(0, 255, 0));
    //}

    //imshow("win", image);
    //waitKey(0);

    //system("pause");
}