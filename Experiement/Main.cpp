#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
using namespace System;
using namespace System::Image;
using namespace System::ML;

inline vector<tuple<Mat, int>> GetImages(const System::String& imageSetPath, int imageLoadMode)
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
    int imageNum = images.size();

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
    int imageNum = images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);

    vector<tuple<vector<FeatureInfo<float>>, vector<FeatureInfo<float>>, vector<int>>> pass = 
        RandomSplit(features, 3);

    vector<FeatureInfo<float>>& evaluationSet = get<0>(pass[2]);
    vector<FeatureInfo<float>>& trainingSet = get<1>(pass[2]);
    vector<int>& pickUpIndexes = get<2>(pass[2]);

    vector<vector<double>> trainingData, evaluationData;
    for (int i = 0; i < trainingSet.size(); i++)
    {
        vector<double> data;
        for (int j = 0; j < trainingSet[i][0].size(); j++)
            data.push_back(trainingSet[i][0][j]);
    }
    for (int i = 0; i < evaluationSet.size(); i++)
    {
        vector<double> data;
        for (int j = 0; j < evaluationSet[i][0].size(); j++)
            data.push_back(evaluationSet[i][0][j]);
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
    pair<double, map<int, double>> precisions = 
        knn.Evaluate(trainingData, trainingLabels, evaluationData, evaluationLabels, 4);
    printf("%f\n", precisions.first);

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

void LocalFeatureCrossValidation(const System::String& imageSetPath, const Feature& feature, 
                                 int wordNum, int sampleNum = 1000000, int fold = 3)
{
    srand(1);
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);
    
    System::String savePath = feature.GetName() + "_knn.out";
    FILE* file = fopen(savePath, "w");
    vector<tuple<vector<FeatureInfo<float>>, vector<FeatureInfo<float>>, vector<int>>> pass = 
        RandomSplit(features, fold);
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
    }

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    fclose(file);
}

void GlobalFeatureCrossValidation(const System::String& imageSetPath, const Feature& feature, 
                                 int wordNum, int sampleNum = 1000000, int fold = 3)
{
    srand(1);
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute " + feature.GetName() + "...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);
    
    System::String savePath = feature.GetName() + "_knn.out";
    FILE* file = fopen(savePath, "w");
    vector<tuple<vector<FeatureInfo<float>>, vector<FeatureInfo<float>>, vector<int>>> pass = 
        RandomSplit(features, fold);
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
    }

    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));
    printf("Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
        Math::StandardDeviation(passResult));

    fclose(file);
}

int main()
{
    //ExtractLocalFeature("oracles_png", HOG(), 500);
    //CrossValidation("oracles_png", HOG(), 500);

    //ExtractLocalFeature("oracles_png", HOOSC(), 1000);
    //CrossValidation("oracles_png", HOOSC(), 1000);

    //ExtractLocalFeature("oracles_png", SC(), 1000);
    //CrossValidation("oracles_png", SC(), 1000);

    //ExtractLocalFeature("oracles_png", SHOG(), 500);
    //CrossValidation("oracles_png", SHOG(), 500);

    //ExtractLocalFeature("oracles_png", RHOOSC(), 1000);
    //CrossValidation("oracles_png", RHOOSC(), 1000);

    //ExtractLocalFeature("oracles_png", RSC(), 1000);
    //CrossValidation("oracles_png", RSC(), 1000);

    //ExtractLocalFeature("oracles_png", ASHOG(), 1000);
    //CrossValidation("oracles_png", ASHOG(), 1000);

    //ExtractLocalFeature("oracles_png", Gabor(), 500);
    //CrossValidation("oracles_png", Gabor(), 500);

    ExtractGlobalFeature("oracles_png", GIST());
}