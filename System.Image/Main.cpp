#include "../System/System.h"
#include "System.Image.h"
using namespace System::Image;

void BHOG(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = HOG().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 500, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

void BHOOSC(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = HOOSC().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 1000, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

void BSC(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = SC().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 1000, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

void BSHOG(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = SHOG().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 500, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

void BRHOOSC(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = RHOOSC().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 1000, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

void BRSC(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = RSC().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 1000, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

void BASHOG(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = ASHOG().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 500, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

void BGabor(const System::String& imageSetPath, const System::String& savePath)
{
    vector<tuple<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
    int imageNum = images.size();

    vector<Feature> features(imageNum);
    printf("Compute Features...\n");
    #pragma omp parallel for
    for (int i = 0; i < imageNum; i++)
        features[i] = Gabor().GetFeatureWithPreprocess(get<0>(images[i]), true);
    
    vector<Feature> trainingFeatures = RandomPickUp(features, features.size() * 2 / 3);
    vector<Descriptor> words = BOV::GetVisualWords(trainingFeatures, 500, 1000000);
    vector<Histogram> freqHistograms = BOV::GetFrequencyHistograms(features, words);

    printf("Writing To File...\n");
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

Mat GetBoundingBox(const Mat& sketchImage)
{
    int minX = sketchImage.cols - 1, maxX = 0,
        minY = sketchImage.rows - 1, maxY = 0;

    for (int i = 0; i < sketchImage.rows; i++)
        for (int j = 0; j < sketchImage.cols; j++)
        {
            if (sketchImage.at<uchar>(i, j))
            {
                minX = min(minX, j);
                maxX = max(maxX, j);
                minY = min(minY, i);
                maxY = max(maxY, i);
            }
        }

    return Mat(sketchImage, Range(minY, maxY + 1), Range(minX, maxX + 1));
}

int main()
{
    BHOG("oracles_png", "bhog_oracles_data");
    BHOOSC("oracles_png", "bhoosc_oracles_data");
    BSC("oracles_png", "bsc_oracles_data");
    BSHOG("oracles_png", "bshog_oracles_data");
    BRHOOSC("oracles_png", "brhoosc_oracles_data");
    BRSC("oracles_png", "brsc_oracles_data");
    BASHOG("oracles_png", "bashog_oracles_data");
    BGabor("oracles_png", "bgabor_oracles_data");

    /*Mat sketchImage = imread("00006.png", CV_LOAD_IMAGE_GRAYSCALE);
    
    Mat revImage = reverse(sketchImage);

    Mat cleanedImage;
    clean(revImage, cleanedImage, 3);

    Mat binaryImage, thinnedImage;
    threshold(cleanedImage, binaryImage, 54, 1, CV_THRESH_BINARY);
    thin(binaryImage, thinnedImage);
    threshold(thinnedImage, revImage, 0.5, 255, CV_THRESH_BINARY);

    Mat boundingBox = GetBoundingBox(revImage);

    Mat squareImage;
    int widthPadding = 0, heightPadding = 0;
    if (boundingBox.rows < boundingBox.cols)
        heightPadding = (boundingBox.cols - boundingBox.rows) / 2;
    else
        widthPadding = (boundingBox.rows - boundingBox.cols) / 2;
    copyMakeBorder(boundingBox, squareImage, heightPadding, heightPadding, 
        widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

    Mat scaledImage;
    resize(squareImage, scaledImage, Size(224, 224));

    Mat paddedImage;
    copyMakeBorder(scaledImage, paddedImage, 16, 16, 16, 16, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
    assert(paddedImage.rows == 256 && paddedImage.cols == 256);

    threshold(paddedImage, binaryImage, 54, 1, CV_THRESH_BINARY);
    thin(binaryImage, thinnedImage);
    threshold(thinnedImage, revImage, 0.5, 255, CV_THRESH_BINARY);

    Mat thre, color;
    threshold(revImage, thre, 128, 1, CV_THRESH_BINARY);
    cvtColor(revImage, color, CV_GRAY2BGR);

    vector<Point2f> corners;
    goodFeaturesToTrack(thre, corners, 100, 0.15, 10);

    for (int i = 0; i < corners.size(); i++)
        circle(color, corners[i], 3, Scalar(0, 255, 255));

    imshow("win", color);
    waitKey(0);*/

    //vector<Edge> edges = EdgeLink(thre);

    //for (auto edge : edges)
    //{
    //    for (auto point : edge)
    //    {
    //        circle(color, point, 1, Scalar(255, 0 ,0));
    //    }

    //    for (auto point : edge)
    //    {
    //        circle(color, point, 1, Scalar(255, 255, 255));
    //    }
    //}
}