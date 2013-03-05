#include "../System/System.h"
#include "../System.Image/System.Image.h"
using namespace System;
using namespace System::Image;

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

void ExtractFeature(const System::String& imageSetPath, const Feature& feature, int wordNum, 
                    int sampleNum = 1000000)
{
    srand(1);
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<FeatureInfo<float>> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        Convert(feature.GetFeatureWithPreprocess(get<0>(images[i]), true), features[i]);
    
    vector<FeatureInfo<float>> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<DescriptorInfo<float>> words = BOV::GetVisualWords(trainingFeatures, wordNum, sampleNum);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Write To File...\n");
    System::String savePath = "oralces_" + feature.GetName();
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

int main()
{
    //ExtractFeature("oracles_png", HOG(), 500);
    ExtractFeature("oracles_png", HOOSC(), 1000);
    /*ExtractFeature("oracles_png", SC(), 1000);
    ExtractFeature("oracles_png", SHOG(), 500);
    ExtractFeature("oracles_png", RHOOSC(), 1000);
    ExtractFeature("oracles_png", RSC(), 1000);
    ExtractFeature("oracles_png", ASHOG(), 1000);
    ExtractFeature("oracles_png", Gabor(), 500);*/
}