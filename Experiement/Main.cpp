#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
using namespace System;
using namespace System::Image;
using namespace System::ML;

inline vector<tuple<Mat, int>> GetImages(const System::String& imageSetPath, 
    int imageLoadMode = CV_LOAD_IMAGE_GRAYSCALE)
{
    System::IO::DirectoryInfo imageSetInfo(imageSetPath);

    vector<System::String> classInfos = imageSetInfo.GetDirectories();
    sort(classInfos.begin(), classInfos.end());

    vector<tuple<Mat, int>> images;
    for (int i = 0; i < classInfos.size(); i++)
    {
        vector<System::String> fileInfos = System::IO::DirectoryInfo(classInfos[i]).GetFiles();
        sort(fileInfos.begin(), fileInfos.end());
        
        for (int j = 0; j < fileInfos.size(); j++)
            images.push_back(make_tuple(imread(fileInfos[j], imageLoadMode), i + 1));
    }

    return images;
}

void ExtractLocalFeature(const System::String& imageSetPath, const Feature& feature, int wordNum, 
                    int sampleNum = 1000000)
{
    srand(1);
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);
    
    vector<DescriptorInfo<float>> words = BOV::GetVisualWords(features, wordNum, sampleNum);

    printf("Compute Frequency Histograms...\n");
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Write To File...\n");
    System::String savePath = feature.GetName() + "_oracles";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < freqHistograms.size(); i++)
    {
        fprintf(file, "%d", get<1>(images[i]));
        for (int j = 0; j < freqHistograms[i].size(); j++)
            fprintf(file, " %d:%f", j + 1, freqHistograms[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);
}

void ExtractGlobalFeature(const System::String& imageSetPath, const Feature& feature)
{
    srand(1);
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);

    printf("Write To File...\n");
    System::String savePath = feature.GetName() + "_oracles";
    FILE* file = fopen(savePath, "w");
    for (int i = 0; i < features.size(); i++)
    {
        fprintf(file, "%d", get<1>(images[i]));
        for (int j = 0; j < features[i][0].size(); j++)
            fprintf(file, " %d:%f", j + 1, features[i][0][j]);
        fprintf(file, "\n");
    }
    fclose(file);
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

tuple<vector<double>, vector<double>> ROC(const vector<double>& distances, 
    const vector<bool>& relevants)
{
    vector<pair<double, int>> distanceAndIndexes;
    for (int i = 0; i < distances.size(); i++)
        distanceAndIndexes.push_back(make_pair(distances[i], i));

    sort(distanceAndIndexes.begin(), distanceAndIndexes.end());

    vector<double> sortedDistances;
    vector<bool> sortedRelevants;
    for (int i = 0; i < distances.size(); i++)
    {
        int index = distanceAndIndexes[i].second;
        sortedDistances.push_back(distances[index]);
        sortedRelevants.push_back(relevants[index]);
    }

    vector<double> positiveDist, negativeDist;
    for (int i = 0; i < sortedRelevants.size(); i++)
    {
        if (sortedRelevants[i])
            positiveDist.push_back(sortedDistances[i]);
        else
            negativeDist.push_back(sortedDistances[i]);
    }

    int numOfCP = 20;
    double firstCP = sortedDistances.front();
    double lastCP = sortedDistances.back();
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

    return make_tuple(DR, FPR);
}

void LocalFeatureCrossValidation(const System::String& imageSetPath, const Feature& feature, 
                                 int wordNum, int sampleNum = 1000000, int fold = 3)
{
    srand(1);
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);
    
    System::String savePath = feature.GetName() + "_oracles_knn.out";
    FILE* file = fopen(savePath, "w");

    vector<tuple<vector<FeatureInfo<float>>, vector<FeatureInfo<float>>, vector<int>>> pass = 
        RandomSplit(features, fold);
    vector<vector<double>> DRs(features.size()), FPRs(features.size());
    vector<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        vector<FeatureInfo<float>>& evaluationSet = get<0>(pass[i]);
        vector<FeatureInfo<float>>& trainingSet = get<1>(pass[i]);
        vector<int>& pickUpIndexes = get<2>(pass[i]);

        vector<DescriptorInfo<float>> words = BOV::GetVisualWords(trainingSet, wordNum, sampleNum);

        printf("Compute Frequency Histograms...\n");
        vector<Histogram> trainingHistograms = BOV::GetFrequencyHistograms(trainingSet, words);
        vector<Histogram> evaluationHistograms = BOV::GetFrequencyHistograms(evaluationSet, words);
        
        vector<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.size() && j == pickUpIndexes[counter])
            {
                evaluationLabels.push_back(get<1>(images[j]));
                counter++;
            }
            else
                trainingLabels.push_back(get<1>(images[j]));
        }

        KNN knn;
        pair<double, map<int, double>> precisions = 
            knn.Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels, 4);
       
        passResult.push_back(precisions.first);
        fprintf(file, "Fold %d: %f\n", i + 1, precisions.first);
        printf("Fold %d: %f\n", i + 1, precisions.first);

        pair<vector<vector<double>>, vector<vector<bool>>> matrix =
            knn.Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels);
        vector<vector<double>>& distances = matrix.first;
        vector<vector<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.size(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = get<0>(roc);
            FPRs[index] = get<1>(roc);
        }
    }

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
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

void GlobalFeatureCrossValidation(const System::String& imageSetPath, const Feature& feature, 
                                  int fold = 3)
{
    srand(1);
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = (int)images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);
    
    /*System::String savePath = feature.GetName() + "_oracles_knn.out";
    FILE* file = fopen(savePath, "w");*/

    vector<tuple<vector<FeatureInfo<float>>, vector<FeatureInfo<float>>, vector<int>>> pass = 
        RandomSplit(features, fold);
    vector<vector<double>> DRs(features.size()), FPRs(features.size());
    //vector<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        vector<FeatureInfo<float>>& evaluationSet = get<0>(pass[i]);
        vector<FeatureInfo<float>>& trainingSet = get<1>(pass[i]);
        vector<int>& pickUpIndexes = get<2>(pass[i]);

        vector<vector<double>> trainingData, evaluationData;
        for (int j = 0; j < trainingSet.size(); j++)
        {
            vector<double> data;
            for (int k = 0; k < trainingSet[j][0].size(); k++)
                data.push_back(trainingSet[j][0][k]);
            trainingData.push_back(data);
        }
        for (int j = 0; j < evaluationSet.size(); j++)
        {
            vector<double> data;
            for (int k = 0; k < evaluationSet[j][0].size(); k++)
                data.push_back(evaluationSet[j][0][k]);
            evaluationData.push_back(data);
        }
        
        vector<int> trainingLabels, evaluationLabels;
        int counter = 0;
        for (int j = 0; j < imageNum; j++)
        {
            if (counter < pickUpIndexes.size() && j == pickUpIndexes[counter])
            {
                evaluationLabels.push_back(get<1>(images[j]));
                counter++;
            }
            else
                trainingLabels.push_back(get<1>(images[j]));
        }

        KNN knn;
        /*pair<double, map<int, double>> precisions = 
            knn.Evaluate(trainingData, trainingLabels, evaluationData, evaluationLabels, 4);
        
        passResult.push_back(precisions.first);
        fprintf(file, "Fold %d: %f\n", i + 1, precisions.first);
        printf("Fold %d: %f\n", i + 1, precisions.first);*/

        pair<vector<vector<double>>, vector<vector<bool>>> matrix =
            knn.Evaluate(trainingData, trainingLabels, evaluationData, evaluationLabels);
        vector<vector<double>>& distances = matrix.first;
        vector<vector<bool>>& relevants = matrix.second;
        for (int j = 0; j < pickUpIndexes.size(); j++)
        {
            int index = pickUpIndexes[j];
            auto roc = ROC(distances[j], relevants[j]);
            DRs[index] = get<0>(roc);
            FPRs[index] = get<1>(roc);
        }
    }

    /*fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    fclose(file);*/

    System::String savePath = feature.GetName() + "_oracles_roc";
    FILE* file = fopen(savePath, "w");
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

int main()
{
    //ExtractLocalFeature("oracles_png", HOG(), 500);
    //LocalFeatureCrossValidation("oracles_png", HOG(), 500);
    //printf("\n");

    //ExtractLocalFeature("oracles_png", HOOSC(), 1000);
    //LocalFeatureCrossValidation("oracles_png", HOOSC(), 1000);
    //printf("\n");

    //ExtractLocalFeature("oracles_png", SC(), 1000);
    //LocalFeatureCrossValidation("oracles_png", SC(), 1000);
    //printf("\n");

    //ExtractLocalFeature("oracles_png", SHOG(), 500);
    //LocalFeatureCrossValidation("oracles_png", SHOG(), 500);
    //printf("\n");

    //ExtractLocalFeature("oracles_png", RHOOSC(), 1000);
    LocalFeatureCrossValidation("oracles_png", RHOOSC(), 1000);
    printf("\n");

    ////ExtractLocalFeature("oracles_png", RSC(), 1000);
    //LocalFeatureCrossValidation("oracles_png", RSC(), 1000);
    //printf("\n");

    ////ExtractLocalFeature("oracles_png", ASHOG(), 1000);
    //LocalFeatureCrossValidation("oracles_png", ASHOG(), 1000);
    //printf("\n");

    //ExtractLocalFeature("oracles_png", Gabor(), 500);
    //LocalFeatureCrossValidation("oracles_png", Gabor(), 500);
    //printf("\n");

    //ExtractGlobalFeature("oracles_png", GIST());
    //GlobalFeatureCrossValidation("oracles_png", GIST());
    //printf("\n");

    //vector<int> width, height;
    //vector<tuple<Mat, int>> result = GetImages("oracles_png");

    //for (auto item : result)
    //{
    //    Mat& image = get<0>(item);
    //    width.push_back(image.cols);
    //    height.push_back(image.rows);
    //}

    //printf("Width -- Average: %f, STD: %f\n", Math::Mean(width), Math::StandardDeviation(width));
    //printf("Height -- Average: %f, STD: %f\n", Math::Mean(height), Math::StandardDeviation(height));



    /*FILE* file = fopen("confusion_matrix", "r");
    vector<int> knn_result(20039), true_result(20039);
    int tmp;
    unordered_map<int, int> dataNumPerClass;

    for (int i = 0; i < 20039; i++)
    {
        fscanf(file, "%d", &tmp);
        tmp--;
        knn_result[i] = tmp;
    }

    for (int i = 0; i < 20039; i++)
    {
        fscanf(file, "%d", &tmp);
        tmp--;
        true_result[i] = tmp;

        dataNumPerClass[tmp]++;
    }

    fclose(file);

    int classNum = dataNumPerClass.size();
    printf("Class Num: %d\n", classNum);

    vector<vector<double>> matrix(classNum);
    for (int i = 0; i < classNum; i++)
        matrix[i] = vector<double>(classNum);

    for (int i = 0; i < 20039; i++)
        matrix[true_result[i]][knn_result[i]]++;

    for (int i = 0; i < classNum; i++)
    {
        int num = dataNumPerClass[i];
        for (int j = 0; j < classNum; j++)
            matrix[i][j] /= num;
    }

    file = fopen("matrix", "w");
    for (int i = 0; i < classNum; i++)
    {
        for (int j = 0; j < classNum; j++)
            fprintf(file, "%f ", matrix[i][j]);
        fprintf(file, "\n");
    }
    fclose(file);*/
}