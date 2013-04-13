#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include <cv.h>
#include <highgui.h>
using namespace System;
using namespace System::IO;
using namespace System::Image;
using namespace System::ML;
using namespace cv;

inline vector<Tuple<Mat, int>> GetImages(const System::String& imageSetPath, 
    int imageLoadMode = CV_LOAD_IMAGE_GRAYSCALE)
{
    DirectoryInfo imageSetInfo(imageSetPath);

    vector<System::String> classInfos = imageSetInfo.GetDirectories();
    sort(classInfos.begin(), classInfos.end());

    vector<Tuple<Mat, int>> images;
    for (int i = 0; i < classInfos.size(); i++)
    {
        vector<System::String> fileInfos = DirectoryInfo(classInfos[i]).GetFiles();
        sort(fileInfos.begin(), fileInfos.end());

        for (int j = 0; j < fileInfos.size(); j++)
            images.push_back(CreateTuple(imread(fileInfos[j], imageLoadMode), i + 1));
    }

    return images;
}

void NormlizeDeviation(vector<Histogram> vecs)
{
    Histogram mean(vecs[0].size());
    double deviation = 0;

    for (size_t i = 0; i < vecs.size(); i++)
        for (size_t j = 0; j < vecs[i].size(); j++)
            mean[j] += vecs[i][j];
    
    for (size_t i = 0; i < mean.size(); i++)
        mean[i] /= vecs.size();

    for (size_t i = 0; i < vecs.size(); i++)
        for (size_t j = 0; j < vecs[i].size(); j++)
            deviation += (vecs[i][j] - mean[j]) * (vecs[i][j] - mean[j]);

    deviation = sqrt(deviation / vecs.size());

    for (size_t i = 0; i < vecs.size(); i++)
        for (size_t j = 0; j < vecs[i].size(); j++)
            vecs[i][j] /= deviation;
}

template<typename LocalFeature>
void LocalFeatureCrossValidation(const System::String& imageSetPath, const LocalFeature& feature, 
                                 int wordNum, bool thinning = false, int sampleNum = 1000000, int fold = 3)
{
    srand(1);
    vector<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<LocalFeature_f> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    LocalFeature machine = feature;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
    {
        Convert(machine.GetFeatureWithPreprocess(images[i].Item1(), thinning), features[i]);
        images[i].Item1().release();
    }

    { // Use a block here to destruct words and freqHistograms immediately.
        printf("Compute Visual Words...\n");
        vector<Word_f> words = BOV::GetVisualWords(features, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

        printf("Write To File...\n");
        System::String savePath = feature.GetName() + "_" + imageSetPath;
        FILE* file = fopen(savePath, "w");
        for (int i = 0; i < freqHistograms.size(); i++)
        {
            fprintf(file, "%d", images[i].Item2());
            for (int j = 0; j < freqHistograms[i].size(); j++)
                fprintf(file, " %d:%f", j + 1, freqHistograms[i][j]);
            fprintf(file, "\n");
        }
        fclose(file);
    }

    vector<Tuple<vector<LocalFeature_f>, vector<LocalFeature_f>, vector<size_t>>> pass = 
        RandomSplit(features, fold);
    vector<vector<double>> DRs(imageNum), FPRs(imageNum);
    vector<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        vector<LocalFeature_f>& evaluationSet = pass[i].Item1();
        vector<LocalFeature_f>& trainingSet = pass[i].Item2();
        vector<size_t>& pickUpIndexes = pass[i].Item3();

        printf("Compute Visual Words...\n");
        vector<Word_f> words = BOV::GetVisualWords(trainingSet, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        vector<Histogram> trainingHistograms = BOV::GetFrequencyHistograms(trainingSet, words);
        vector<Histogram> evaluationHistograms = BOV::GetFrequencyHistograms(evaluationSet, words);
        
        vector<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.size() && j == pickUpIndexes[counter])
            {
                evaluationLabels.push_back(images[j].Item2());
                counter++;
            }
            else
                trainingLabels.push_back(images[j].Item2());
        }

        KNN<Histogram> knn;
        pair<double, map<int, double>> precisions = 
            knn.Evaluate(4, trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels);
       
        passResult.push_back(precisions.first);
        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);

        pair<vector<vector<double>>, vector<vector<bool>>> matrix =
            knn.Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels);
        vector<vector<double>>& distances = matrix.first;
        vector<vector<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.size(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = roc.Item1();
            FPRs[index] = roc.Item2();
        }
    }

    System::String savePath = feature.GetName() + "_" + imageSetPath + "_knn.out";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < passResult.size(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    fclose(file);

    savePath = feature.GetName() + "_" + imageSetPath + "_roc";
    file = fopen(savePath, "w");
    for (int i = 0; i < DRs.size(); i++)
    {
        for (int j = 0; j < DRs[i].size(); j++)
            fprintf(file, "%f ", DRs[i][j]);
        fprintf(file, "\n");

        for (int j = 0; j < FPRs[i].size(); j++)
            fprintf(file, "%f ", FPRs[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);
}

template<typename GlobalFeature>
void GlobalFeatureCrossValidation(const System::String& imageSetPath, const GlobalFeature& feature, 
                                  bool thinning = false, int fold = 3)
{
    srand(1);
    vector<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<GlobalFeature_f> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    GlobalFeature machine = feature;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
    {
        Convert(machine.GetFeatureWithPreprocess(images[i].Item1(), thinning), features[i]);
        images[i].Item1().release();
    }

    printf("Write To File...\n");
    System::String savePath = feature.GetName() + "_" + imageSetPath;
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < features.size(); i++)
    {
        fprintf(file, "%d", images[i].Item2());
        for (int j = 0; j < features[i].size(); j++)
            fprintf(file, " %d:%f", j + 1, features[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);

    vector<Tuple<vector<GlobalFeature_f>, vector<GlobalFeature_f>, vector<size_t>>> pass = 
        RandomSplit(features, fold);
    vector<vector<double>> DRs(imageNum), FPRs(imageNum);
    vector<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        vector<GlobalFeature_f>& evaluationSet = pass[i].Item1();
        vector<GlobalFeature_f>& trainingSet = pass[i].Item2();
        vector<size_t>& pickUpIndexes = pass[i].Item3();

        vector<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.size() && j == pickUpIndexes[counter])
            {
                evaluationLabels.push_back(images[j].Item2());
                counter++;
            }
            else
                trainingLabels.push_back(images[j].Item2());
        }

        KNN<GlobalFeature_f> knn;
        pair<double, map<int, double>> precisions = 
            knn.Evaluate(4, trainingSet, trainingLabels, evaluationSet, evaluationLabels);

        passResult.push_back(precisions.first);
        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);

        pair<vector<vector<double>>, vector<vector<bool>>> matrix =
            knn.Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels);
        vector<vector<double>>& distances = matrix.first;
        vector<vector<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.size(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = roc.Item1();
            FPRs[index] = roc.Item2();
        }
    }

    savePath = feature.GetName() + "_" + imageSetPath + "_knn.out";
    file = fopen(savePath, "w");
    for (int i = 0; i < passResult.size(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    fclose(file);

    savePath = feature.GetName() + "_" + imageSetPath + "_roc";
    file = fopen(savePath, "w");
    for (int i = 0; i < DRs.size(); i++)
    {
        for (int j = 0; j < DRs[i].size(); j++)
            fprintf(file, "%f ", DRs[i][j]);
        fprintf(file, "\n");

        for (int j = 0; j < FPRs[i].size(); j++)
            fprintf(file, "%f ", FPRs[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);
}

template<typename EdgeMatching>
void EdgeMatchingCrossValidation(const System::String& imageSetPath, const EdgeMatching& matching,
                                 bool thinning = false, int fold = 3)
{
    srand(1);
    vector<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<EdgeMatching::Info> transforms(imageNum);
    printf("Compute " + matching.GetName() + "...\n");
    EdgeMatching machine = matching;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
    {
        transforms[i] = machine.GetFeatureWithPreprocess(images[i].Item1(), thinning);
        images[i].Item1().release();
    }

    Mat distanceMatrix(transforms.size(), transforms.size() - 1, CV_64F);
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
    System::String savePath = matching.GetName() + "_" + imageSetPath + "_matrix";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < transforms.size(); i++)
    {
        fprintf(file, "%d", images[i].Item2());
        for (int j = 0; j < transforms.size(); j++)
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

    vector<Tuple<vector<EdgeMatching::Info>, vector<EdgeMatching::Info>, vector<size_t>>> pass = 
        RandomSplit(transforms, fold);
    vector<vector<double>> DRs(imageNum), FPRs(imageNum);
    vector<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        vector<EdgeMatching::Info>& evaluationSet = pass[i].Item1();
        vector<EdgeMatching::Info>& trainingSet = pass[i].Item2();
        vector<size_t>& pickUpIndexes = pass[i].Item3();

        vector<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.size() && j == pickUpIndexes[counter])
            {
                evaluationLabels.push_back(images[j].Item2());
                counter++;
            }
            else
                trainingLabels.push_back(images[j].Item2());
        }

        KNN<EdgeMatching::Info> knn;
        pair<double, map<int, double>> precisions = knn.Evaluate(
            4, trainingSet, trainingLabels, evaluationSet, evaluationLabels, EdgeMatching::GetDistance);

        passResult.push_back(precisions.first);
        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);

        pair<vector<vector<double>>, vector<vector<bool>>> matrix = knn.Evaluate(
            trainingSet, trainingLabels, evaluationSet, evaluationLabels, EdgeMatching::GetDistance);
        vector<vector<double>>& distances = matrix.first;
        vector<vector<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.size(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = roc.Item1();
            FPRs[index] = roc.Item2();
        }
    }

    savePath = matching.GetName() + "_" + imageSetPath + "_knn.out";
    file = fopen(savePath, "w");
    for (int i = 0; i < passResult.size(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    fclose(file);

    savePath = matching.GetName() + "_" + imageSetPath + "_roc";
    file = fopen(savePath, "w");
    for (int i = 0; i < DRs.size(); i++)
    {
        for (int j = 0; j < DRs[i].size(); j++)
            fprintf(file, "%f ", DRs[i][j]);
        fprintf(file, "\n");

        for (int j = 0; j < FPRs[i].size(); j++)
            fprintf(file, "%f ", FPRs[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);
}

vector<double> Boosting(const vector<Histogram>& data, const vector<int>& labels)
{
    vector<Tuple<vector<Histogram>, vector<Histogram>, vector<size_t>>> pass = 
        RandomSplit(data, 3);

    int histSize = data[0].size();
    vector<double> weights(histSize);

    for (int k = 0; k < histSize; k++)
    {
        vector<Histogram>& evaluationSet = pass[0].Item1();
        vector<Histogram>& trainingSet = pass[0].Item2();
        vector<size_t>& pickUpIndexes = pass[0].Item3();

        vector<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < data.size(); j++)
        {
            if (counter < pickUpIndexes.size() && j == pickUpIndexes[counter])
            {
                evaluationLabels.push_back(labels[j]);
                counter++;
            }
            else
                trainingLabels.push_back(labels[j]);
        }

        vector<Histogram> evaluationData(evaluationSet.size()), trainingData(trainingSet.size());
        for (int j = 0; j < evaluationSet.size(); j++)
            evaluationData[j].push_back(evaluationSet[j][k]);
        for (int j = 0; j < trainingSet.size(); j++)
            trainingData[j].push_back(trainingSet[j][k]);

        KNN<Histogram> knn;
        pair<double, map<int, double>> precisions = 
            knn.Evaluate(4, trainingData, trainingLabels, evaluationData, evaluationLabels);

        weights[k] = precisions.first;
    }

    return weights;
}

template<typename T, typename U>
vector<T> operator*(const vector<T>& vec, const U& factor)
{
    vector<T> result(vec.size());

    for (size_t i = 0; i < vec.size(); i++)
        result[i] = vec[i] * factor;

    return result;
}

vector<vector<LocalFeature_f>> DivideLocalFeatures(const vector<LocalFeature_f> features)
{
    int blockSize = 128;

    vector<Point> wordCenters;
    for (int i = 1; i <= 3; i++)
        for (int j = 1; j <= 3; j++)
            wordCenters.push_back(Point(i * 64, j * 64));

    vector<Point> descPoints = SampleOnGrid(256, 256, 28);
    assert(descPoints.size() == features[0].size());

    vector<vector<LocalFeature_f>> parts(wordCenters.size());
    for (int i = 0; i < features.size(); i++)
    {
        vector<LocalFeature_f> tmp(wordCenters.size());

        for (int j = 0; j < wordCenters.size(); j++)
        {
            int top = wordCenters[j].y - blockSize / 2,
                bottom = wordCenters[j].y + blockSize / 2,
                left = wordCenters[j].x - blockSize / 2,
                right = wordCenters[j].x + blockSize / 2;

            for (int k = 0; k < descPoints.size(); k++)
            {
                if (top <= descPoints[k].y && descPoints[k].y < bottom &&
                    left <= descPoints[k].x && descPoints[k].x < right)
                    tmp[j].push_back(features[i][k]);
            }
        }

        for (int j = 0; j < wordCenters.size(); j++)
            parts[j].push_back(tmp[j]);
    }

    return parts;
}

template<typename LocalFeature>
void LocalFeatureTest(const System::String& imageSetPath, const LocalFeature& feature, 
                                 int wordNum, int sampleNum = 1000000, int fold = 3)
{
    srand(1);
    vector<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<LocalFeature_f> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    LocalFeature machine = feature;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
        Convert(machine.GetFeatureWithPreprocess(images[i].Item1(), true), features[i]);

    vector<Tuple<vector<LocalFeature_f>, vector<LocalFeature_f>, vector<size_t>>> pass = 
        RandomSplit(features, fold);
    vector<double> passResult;
    map<int, double> ap;
    map<int, int> count;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        vector<LocalFeature_f>& evaluationSet = pass[i].Item1();
        vector<LocalFeature_f>& trainingSet = pass[i].Item2();
        vector<size_t>& pickUpIndexes = pass[i].Item3();

        /*vector<LocalFeature_f> evaluationSet1(evaluationSet.size());
        vector<LocalFeature_f> evaluationSet2(evaluationSet.size());
        for (int i = 0; i < evaluationSet.size(); i++)
        {
            evaluationSet1[i].push_back(evaluationSet[i].begin(), 
                evaluationSet[i].begin() + evaluationSet[i].size() / 2);
            evaluationSet2[i].push_back(evaluationSet[i].begin() + evaluationSet[i].size() / 2, 
                evaluationSet[i].end());
        }

        vector<LocalFeature_f> trainingSet1(trainingSet.size());
        vector<LocalFeature_f> trainingSet2(trainingSet.size());
        for (int i = 0; i < trainingSet.size(); i++)
        {
            trainingSet1[i].push_back(trainingSet[i].begin(), 
                trainingSet[i].begin() + trainingSet[i].size() / 2);
            trainingSet2[i].push_back(trainingSet[i].begin() + trainingSet[i].size() / 2, 
                trainingSet[i].end());
        }*/

        printf("Compute Visual Words...\n");
        vector<Word_f> words = BOV::GetVisualWords(trainingSet, wordNum, sampleNum);

        //vector<Word_f> words1 = BOV::GetVisualWords(trainingSet1, wordNum, sampleNum);
        //vector<Word_f> words2 = BOV::GetVisualWords(trainingSet2, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        vector<Histogram> trainingHistograms = BOV::GetFrequencyHistograms(trainingSet, words) * 9;
        vector<Histogram> evaluationHistograms = BOV::GetFrequencyHistograms(evaluationSet, words) * 9;

        vector<vector<LocalFeature_f>> parts = DivideLocalFeatures(trainingSet);
        for (int j = 0; j < parts.size(); j++)
        {
            vector<Histogram> result = BOV::GetFrequencyHistograms(parts[j], words);

            assert(result.size() == trainingHistograms.size());
            for (int k = 0; k < result.size(); k++)
                trainingHistograms[k].push_back(result[k]);
        }

        parts = DivideLocalFeatures(evaluationSet);
        for (int j = 0; j < parts.size(); j++)
        {
            vector<Histogram> result = BOV::GetFrequencyHistograms(parts[j], words);

            assert(result.size() == evaluationHistograms.size());
            for (int k = 0; k < result.size(); k++)
                evaluationHistograms[k].push_back(result[k]);
        }

        assert(trainingHistograms[0].size() == wordNum * 10 &&
            evaluationHistograms[0].size() == wordNum * 10);

        /*vector<Histogram> trainingHistograms1 = BOV::GetFrequencyHistograms(trainingSet1, words1);
        vector<Histogram> trainingHistograms2 = BOV::GetFrequencyHistograms(trainingSet2, words2);

        vector<Histogram> evaluationHistograms1 = BOV::GetFrequencyHistograms(evaluationSet1, words1);
        vector<Histogram> evaluationHistograms2 = BOV::GetFrequencyHistograms(evaluationSet2, words2);

        vector<Histogram> trainingHistograms(trainingSet.size()), evaluationHistograms(evaluationSet.size());

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

        vector<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int k = 0; k < imageNum; k++)
        {
            if (counter < pickUpIndexes.size() && k == pickUpIndexes[counter])
            {
                evaluationLabels.push_back(images[k].Item2());
                counter++;
            }
            else
                trainingLabels.push_back(images[k].Item2());
        }

        //vector<double> weights = Boosting(trainingHistograms, trainingLabels);
        //for (int k = 0; k < weights.size(); k++)
        //{
        //    for (int i = 0; i < trainingHistograms.size(); i++)
        //        trainingHistograms[i][k] *= weights[k];

        //    for (int i = 0; i < evaluationHistograms.size(); i++)
        //        evaluationHistograms[i][k] *= weights[k];
        //}

        //printf("Perform LDA...\n");
        //pair<vector<Histogram>, vector<Histogram>> result = LDAOperator::ComputeLDA(
        //    trainingHistograms, trainingLabels, 1000, evaluationHistograms);
        //trainingHistograms = result.first;
        //evaluationHistograms = result.second;

        KNN<Histogram> knn;
        pair<double, map<int, double>> precisions = 
            knn.Evaluate(4, trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels);

        passResult.push_back(precisions.first);
        for (auto item : precisions.second)
        {
            ap[item.first] += item.second;
            count[item.first]++;
        }

        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);
    }

    System::String savePath = feature.GetName() + "_oracles_knn.out";
    FILE* file = fopen(savePath, "w");

    for (int i = 0; i < passResult.size(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    for (auto item : ap)
        fprintf(file, "Class %d: %f\n", item.first, item.second / count[item.first]);

    fclose(file);
}

void Batch(const System::String& imageSetPath, bool thinning = false)
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
    //LocalFeatureCrossValidation("sketches", HOG(), 500);
    //printf("\n");

    //GlobalFeatureCrossValidation("oracles_png", GHOG());
    //printf("\n");

    //EdgeMatchingCrossValidation("oracles", OCM(), true);
    //printf("\n");

    //LocalFeatureTest("oracles_png", Test(), 1500);
    //printf("\n");

    //Batch("sketches", false);
    //Batch("oracles", true);

    //Mat trans = getRotationMatrix2D(Point(image.rows / 2, image.cols / 2), -20, 1);
    //warpAffine(image, image, trans, image.size());
    //threshold(image, image, 0.1, 1, CV_THRESH_BINARY);
    //thin(image, image);
    //threshold(image, image, 0.1, 255, CV_THRESH_BINARY);

    /*vector<Point> points = GetEdgels(image);
    vector<int> xCount(image.cols);

    for (auto point : points)
        xCount[point.x]++;

    for (int i = 1; i < xCount.size(); i++)
        xCount[i] += xCount[i - 1];

    int blockNum = 4;
    double xStep = xCount.back() / (double)blockNum;

    vector<int> xSep(blockNum + 1);
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

        vector<int> yCount(image.rows);
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