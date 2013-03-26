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

inline vector<double> linspace(double start, double end, int pointNum)
{
    double size = (end - start) / (pointNum - 1);

    vector<double> result;
    result.push_back(start);
    for (int i = 1; i < pointNum - 1; i++)
        result.push_back(result[i - 1] + size);
    result.push_back(end);

    assert(result.size() == pointNum);
    return result;
}

Tuple<vector<double>, vector<double>> ROC(const vector<double>& distances, 
    const vector<bool>& relevants)
{
    vector<double> positiveDist, negativeDist;
    for (int i = 0; i < relevants.size(); i++)
    {
        if (relevants[i])
            positiveDist.push_back(distances[i]);
        else
            negativeDist.push_back(distances[i]);
    }

    int numOfCP = 20;
    double firstCP = Math::Min(distances);
    double lastCP = Math::Max(distances);
    vector<double> plot = linspace(firstCP, lastCP, numOfCP);

    vector<double> TP(numOfCP), FP(numOfCP), TN(numOfCP), FN(numOfCP);
    for (int i = 0; i < numOfCP; i++)
    {
        for (auto item : positiveDist)
            if (item <= plot[i])
                TP[i]++;

        for (auto item : negativeDist)
            if (item <= plot[i])
                FP[i]++;

        for (auto item : positiveDist)
            if (item > plot[i])
                FN[i]++;

        for (auto item : negativeDist)
            if (item > plot[i])
                TN[i]++;

        assert(TP[i] + FN[i] == positiveDist.size() && FP[i] + TN[i] == negativeDist.size());
    }

    vector<double> DR, FPR;
    for (int i = 0; i < numOfCP; i++)
    {
        DR.push_back(TP[i] / (TP[i] + FN[i]));
        FPR.push_back(FP[i] / (FP[i] + TN[i]));
    }

    return CreateTuple(DR, FPR);
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

    { // Use a block here to destruct words and freqHistograms immediately.
        printf("Compute Visual Words...\n");
        vector<Word_f> words = BOV::GetVisualWords(features, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

        printf("Write To File...\n");
        System::String savePath = feature.GetName() + "_oracles";
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

    System::String savePath = feature.GetName() + "_oracles_knn.out";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < passResult.size(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    fclose(file);

    savePath = feature.GetName() + "_oracles_roc";
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
                                  int fold = 3)
{
    srand(1);
    vector<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<GlobalFeature_f> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    GlobalFeature machine = feature;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
        Convert(machine.GetFeatureWithPreprocess(images[i].Item1(), true), features[i]);

    printf("Write To File...\n");
    System::String savePath = feature.GetName() + "_oracles";
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

    savePath = feature.GetName() + "_oracles_knn.out";
    file = fopen(savePath, "w");
    for (int i = 0; i < passResult.size(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    fclose(file);

    savePath = feature.GetName() + "_oracles_roc";
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
                                 int fold = 3)
{
    srand(1);
    vector<Tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<EdgeMatching::Info> transforms(imageNum);
    printf("Compute " + matching.GetName() + "...\n");
    EdgeMatching machine = matching;
    #pragma omp parallel for private(machine)
    for (int i = 0; i < imageNum; i++)
        transforms[i] = machine.GetFeatureWithPreprocess(images[i].Item1(), true);

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

    System::String savePath = matching.GetName() + "_oracles_knn.out";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < passResult.size(); i++)
        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    fclose(file);

    savePath = matching.GetName() + "_oracles_roc";
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

        vector<LocalFeature_f> evaluationSet1(evaluationSet.size());
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
        }

        printf("Compute Visual Words...\n");
        //vector<Word_f> words = BOV::GetVisualWords(trainingSet, wordNum, sampleNum);

        vector<Word_f> words1 = BOV::GetVisualWords(trainingSet1, wordNum, sampleNum);
        vector<Word_f> words2 = BOV::GetVisualWords(trainingSet2, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        //vector<Histogram> trainingHistograms = BOV::GetFrequencyHistograms(trainingSet, words);
        //vector<Histogram> evaluationHistograms = BOV::GetFrequencyHistograms(evaluationSet, words);

        vector<Histogram> trainingHistograms1 = BOV::GetFrequencyHistograms(trainingSet1, words1);
        vector<Histogram> trainingHistograms2 = BOV::GetFrequencyHistograms(trainingSet2, words2);

        vector<Histogram> evaluationHistograms1 = BOV::GetFrequencyHistograms(evaluationSet1, words1);
        vector<Histogram> evaluationHistograms2 = BOV::GetFrequencyHistograms(evaluationSet2, words2);

        vector<Histogram> trainingHistograms(trainingSet.size()), evaluationHistograms(evaluationSet.size());

        for (int i = 0; i < trainingSet.size(); i++)
        {
            NormlizeDeviation(trainingHistograms1);
            trainingHistograms[i].push_back(trainingHistograms1[i].begin(), trainingHistograms1[i].end());

            NormlizeDeviation(trainingHistograms2);
            trainingHistograms[i].push_back(trainingHistograms2[i].begin(), trainingHistograms2[i].end());
        }

        for (int i = 0; i < evaluationSet.size(); i++)
        {
            NormlizeDeviation(evaluationHistograms1);
            evaluationHistograms[i].push_back(evaluationHistograms1[i].begin(), evaluationHistograms1[i].end());

            NormlizeDeviation(evaluationHistograms2);
            evaluationHistograms[i].push_back(evaluationHistograms2[i].begin(), evaluationHistograms2[i].end());
        }

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

void Batch()
{
    LocalFeatureCrossValidation("oracles_png", HOG(), 500);
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", SHOG(), 500);
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", LogSHOG(), 500);
    printf("\n");

    GlobalFeatureCrossValidation("oracles_png", GHOG());
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", HOOSC(), 1000);
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", RHOOSC(), 1000);
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", SC(), 1000);
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", PSC(), 1000);
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", RSC(), 1000);
    printf("\n");

    LocalFeatureCrossValidation("oracles_png", Gabor(), 500);
    printf("\n");

    GlobalFeatureCrossValidation("oracles_png", GIST());
    printf("\n");

    EdgeMatchingCrossValidation("oracles_png", CM());
    printf("\n");

    EdgeMatchingCrossValidation("oracles_png", OCM());
    printf("\n");

    EdgeMatchingCrossValidation("oracles_png", Hitmap());
    printf("\n");
}

int main()
{
    //LocalFeatureCrossValidation("oracles_png", Test(), 1000);
    //printf("\n");

    //GlobalFeatureCrossValidation("oracles_png", GHOG());
    //printf("\n");

    LocalFeatureTest("oracles_png", Test(), 1500);
    printf("\n");

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

    system("pause");
}