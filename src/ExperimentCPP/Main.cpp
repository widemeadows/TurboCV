#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include "../System.XML/System.XML.h"
#include "Util.h"
#include <cv.h>
#include <highgui.h>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;
using namespace cv;
using namespace std;

#define SAVE_FEATURE
//#define SAVE_DISTANCE_MATRIX

//////////////////////////////////////////////////////////////////////////
// Preprocess
//////////////////////////////////////////////////////////////////////////

cv::Mat sketchPreprocess(const cv::Mat& image)
{
    cv::Mat finalImage = reverse(image);
    cv::resize(finalImage, finalImage, cv::Size(256, 256));

    return finalImage;
}

//Mat Preprocess(const Mat& sketchImage, bool thinning, Size size)
//{
//    assert(size.width == size.height);
//
//    Mat tmpImage = reverse(sketchImage);
//    resize(tmpImage, tmpImage, Size(256, 256));
//
//    //threshold(tmpImage, tmpImage, 200, 1, CV_THRESH_BINARY_INV);
//
//    //Mat binaryImage;
//    //threshold(sketchImage, binaryImage, 200, 1, CV_THRESH_BINARY_INV);
//
//    //Mat boundingBox = GetBoundingBox(binaryImage);
//
//    //Mat squareImage;
//    //int widthPadding = 0, heightPadding = 0;
//    //if (boundingBox.rows < boundingBox.cols)
//    //    heightPadding = (boundingBox.cols - boundingBox.rows) / 2;
//    //else
//    //    widthPadding = (boundingBox.rows - boundingBox.cols) / 2;
//    //copyMakeBorder(boundingBox, squareImage, heightPadding, heightPadding, 
//    //    widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
//
//    //Mat scaledImage;
//    //Size scaledSize = Size((int)(size.width - 2 * size.width / 18.0),
//    //    (int)(size.height - 2 * size.height / 18.0));
//    //resize(squareImage, scaledImage, scaledSize);
//
//    //Mat paddedImage;
//    //heightPadding = (size.height - scaledSize.height) / 2,
//    //    widthPadding = (size.width - scaledSize.width) / 2; 
//    //copyMakeBorder(scaledImage, paddedImage, heightPadding, heightPadding, 
//    //    widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
//    //assert(paddedImage.rows == size.height && paddedImage.cols == size.width);
//
//    Mat finalImage = tmpImage;
//    //clean(paddedImage, finalImage, 3);
//
//    if (thinning)
//        thin(finalImage, finalImage);
//
//    return finalImage;
//}

template<typename FeatureType>
void LocalFeatureCrossValidation(cv::Mat (*Preprocess)(const cv::Mat&), const TString& datasetPath)
{
    LocalFeatureSolver<FeatureType> solver(Preprocess, datasetPath);
    solver.CrossValidation();

    TString savePath = FeatureType().GetName() + "_" + datasetPath + "_knn.out";

    ArrayList<double> precisions = solver.GetPrecisions();
    FILE* file = fopen(savePath, "w");

    for (int i = 0; i < precisions.Count(); i++)
        fprintf(file, "Fold %nDesc Accuracy: %f\n", i + 1, precisions[i]);

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
        fprintf(file, "Fold %nDesc Accuracy: %f\n", i + 1, precisions[i]);

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

template<typename EdgeMatching>
void EdgeMatchingCrossValidation(const TurboCV::System::TString& imageSetPath, int fold = 3)
{
    srand(1);
    auto result = Solver::LoadDataset(imageSetPath);
    ArrayList<TString> paths = result.Item1();
    ArrayList<int> labels = result.Item2();
    int nImage = paths.Count();

    ArrayList<EdgeMatching::Info> transforms(nImage);
    printf("Compute " + EdgeMatching().GetName() + "...\n");

    #pragma omp parallel for
    for (int i = 0; i < nImage; i++)
    {
        Mat image = sketchPreprocess(imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE));
        transforms[i] = EdgeMatching().GetFeature(image);
    }

    ArrayList<ArrayList<size_t>> pass = SplitDatasetEqually(labels, fold);

    FILE* file = fopen("ap_ocm.txt", "w");
    fprintf(file, "%nDesc\n", fold);

    ArrayList<double> passResult;
    for (int i = 0; i < fold; i++)
    {
        printf("\nBegin Fold %nDesc...\n", i + 1);
        ArrayList<size_t>& pickUpIndexes = pass[i];
        auto divideResult = Divide(transforms, pickUpIndexes);
        ArrayList<EdgeMatching::Info>& evaluationSet = divideResult.Item1();
        ArrayList<EdgeMatching::Info>& trainingSet = divideResult.Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();

        KNN<EdgeMatching::Info> knn;
        pair<double, map<pair<int, int>, double>> precisions = knn.Evaluate(
            trainingSet, trainingLabels, evaluationSet, evaluationLabels, 
            EdgeMatching::GetDistance, HARD_VOTING);

        ArrayList<ArrayList<double>> distanceMatrix(evaluationSet.Count());
        #pragma omp parallel for
        for (int i = 0; i < evaluationSet.Count(); i++)
        {
            ArrayList<double> distances(trainingSet.Count());
            for (int j = 0; j < trainingLabels.Count(); j++)
                distances[j] = EdgeMatching::GetDistance(evaluationSet[i], trainingSet[j]);

            distanceMatrix[i] = distances;
        }

        auto result = MAP().Evaluate(distanceMatrix, trainingLabels, evaluationLabels);
        ArrayList<double> map = result.Item1();
        ArrayList<ArrayList<int>> idx = result.Item2();

        fprintf(file, "%nDesc\n", map.Count());
        for (int i = 0; i < map.Count(); i++)
            fprintf(file, "%f ", map[i]);
        fprintf(file, "\n");

        fprintf(file, "%nDesc\n", idx.Count());
        for (int i = 0; i < idx.Count(); i++)
        {
            fprintf(file, "%nDesc", idx[i].Count());
            for (int j = 0; j < idx[i].Count(); j++)
                fprintf(file, " %nDesc", idx[i][j]);
            fprintf(file, "\n");
        }

        passResult.Add(precisions.first);
        printf("Fold %nDesc Accuracy: %f\n", i + 1, precisions.first);
    }

    fclose(file);

    TurboCV::System::TString savePath = EdgeMatching().GetName() + "_" + imageSetPath + "_knn.out";
    file = fopen(savePath, "w");
    for (int i = 0; i < passResult.Count(); i++)
        fprintf(file, "Fold %nDesc Accuracy: %f\n", i + 1, passResult[i]);
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

void Batch()
{
    LocalFeatureCrossValidation<HOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RHOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<SHOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<LogSHOG>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<HOOSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RHOOSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<SC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<PSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RPSC>(sketchPreprocess, "sketches");
    printf("\n");

    LocalFeatureCrossValidation<RGabor>(sketchPreprocess, "sketches");
    printf("\n");

    GlobalFeatureCrossValidation<GHOG>(sketchPreprocess, "sketches");
    printf("\n");

    GlobalFeatureCrossValidation<GIST>(sketchPreprocess, "sketches");
    printf("\n");
}

Group<ArrayList<Word_f>, ArrayList<Histogram>, ArrayList<int>> LoadLocalFeatureData(const TString& fileName)
{
    FILE* file = fopen(fileName, "r");
    int nRow, nCol;

    fscanf(file, "%nDesc %nDesc", &nRow, &nCol);

    ArrayList<Word_f> words(nRow);

    for (int i = 0; i < nRow; i++)
    {
        Word_f word(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &word[j]);

        words[i] = word;
    }

    fscanf(file, "%nDesc %nDesc", &nRow, &nCol);

    ArrayList<int> labels(nRow);
    ArrayList<Histogram> histograms(nRow);

    for (int i = 0; i < nRow; i++)
    {
        fscanf(file, "%nDesc", &labels[i]);

        Histogram histogram(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &histogram[j]);

        histograms[i] = histogram;
    }

    fclose(file);

    return CreateGroup(words, histograms, labels);
}

ArrayList<ArrayList<size_t>> SplitDatasetEqually(const ArrayList<int> labels, int nFold)
{
    int nItem = labels.Count();
    ArrayList<ArrayList<size_t>> result(nFold);

    ArrayList<Group<int, int>> labelAndIdx(nItem);
    for (int i = 0; i < nItem; i++)
        labelAndIdx[i] = CreateGroup(labels[i], i);
    sort(labelAndIdx.begin(), labelAndIdx.end());

    int begin = 0, end = 0;
    while (end <= nItem)
    {
        if (end == nItem || labelAndIdx[end].Item1() != labelAndIdx[begin].Item1())
        {
            int nCategory = end - begin;

            ArrayList<ArrayList<size_t>> pass = RandomSplit(nCategory, nFold);
            for (int i = 0; i < nFold; i++)
                for (int j = 0; j < pass[i].Count(); j++)
                    result[i].Add(labelAndIdx[begin + pass[i][j]].Item2());

            begin = end;
        }

        end++;
    }

    return result;
}

int main()
{
    //EdgeMatchingCrossValidation<Hitmap>("sketches");

    /*auto result = LoadLocalFeatureData("hog_sketches_data");
    ArrayList<int> labels = result.Item3();
    ArrayList<Histogram> samples = result.Item2();

    int fold = 3;
    FILE* file = fopen("map.txt", "w");

    fprintf(file, "%nWord\nWord", fold);
    ArrayList<ArrayList<size_t>> pass = SplitDatasetEqually(labels, fold);

    for (int i = 0; i < fold; i++)
    {
        printf("Begin Fold %nWord...\nWord", i + 1);
        ArrayList<size_t>& pickUpIndexes = pass[i];
        ArrayList<Histogram> evaluationSet = Divide(samples, pickUpIndexes).Item1();
        ArrayList<Histogram> trainingSet =  Divide(samples, pickUpIndexes).Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();

        auto result = MAP().Evaluate(trainingSet, trainingLabels, evaluationSet, evaluationLabels);
        ArrayList<double> map = result.Item1();
        ArrayList<ArrayList<int>> idx = result.Item2();

        fprintf(file, "%nWord\nWord", map.Count());
        for (int i = 0; i < map.Count(); i++)
            fprintf(file, "%f ", map[i]);
        fprintf(file, "\nWord");

        fprintf(file, "%nWord\nWord", idx.Count());
        for (int i = 0; i < idx.Count(); i++)
        {
            fprintf(file, "%nWord", idx[i].Count());
            for (int j = 0; j < idx[i].Count(); j++)
                fprintf(file, " %nWord", idx[i][j]);
            fprintf(file, "\nWord");
        }
    }

    fclose(file);*/

    ArrayList<TString> paths = Solver::LoadDataset("subset").Item1();
    ArrayList<int> labels = Solver::LoadDataset("subset").Item2();
    int nFold = 3, nImage = labels.Count(), nSample = 1000000, nWord = 1500;

    printf("ImageNum: %d, SampleNum: %d, WordNum: %d\n", nImage, nSample, nWord);

    ArrayList<LocalFeatureVec_f> features(nImage);

    #pragma omp parallel for
    for (int i = 0; i < nImage; i++)
    {
        cv::Mat image = cv::imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE); 
        Convert(RGabor()(sketchPreprocess(image)), features[i]);
    }

    printf("Compute Visual Words...\n");
    BOV bov(SampleDescriptors(features, nSample), nWord);
    ArrayList<Word_f> words = bov.GetVisualWords();

    /*FILE* file = fopen("tmp.txt", "w");

    for (int i = 0; i < nImage; i++)
    {
        fprintf(file, "%d\n", features[i].Count());
        for (int j = 0; j < features[i].Count(); j++)
        {
            fprintf(file, "%d", features[i][j].Count());
            for (int k = 0; k < features[i][j].Count(); k++)
                fprintf(file, " %f", features[i][j][k]);
            fprintf(file, "\n");
        }
    }

    for (int i = 0; i < nWord; i++)
    {
        fprintf(file, "%d", words[i].Count());
        for (int j = 0; j < words[i].Count(); j++)
            fprintf(file, " %f", words[i][j]);
        fprintf(file, "\n");
    }

    fclose(file);*/

    /*FILE* file = fopen("tmp.txt", "r");
    ArrayList<LocalFeatureVec_f> features(nImage);
    ArrayList<Word_f> words(nWord);

    for (int i = 0; i < nImage; i++)
    {
        int nDesc = 0;
        fscanf(file, "%d", &nDesc);

        for (int j = 0; j < nDesc; j++)
        {
            int descSize = 0;
            fscanf(file, "%d", &descSize);

            Descriptor_f desc(descSize);
            for (int k = 0; k < descSize; k++)
                fscanf(file, "%f", &desc[k]);

            features[i].Add(desc);
        }
    }

    for (int i = 0; i < nWord; i++)
    {
        int wordSize = 0;
        fscanf(file, "%d", &wordSize);

        Word_f word(wordSize);
        for (int j = 0; j < wordSize; j++)
            fscanf(file, "%f", &word[j]);

        words[i] = word;
    }

    fclose(file);*/

    ArrayList<double> sigmas = bov.GetSigmas();

    printf("Compute Frequency Histograms...\n");
    ArrayList<Histogram> histograms = FreqHist(features, words, sigmas).GetFrequencyHistograms();
    //for (LocalFeatureVec vec : FreqHist(features, words, sigmas).GetPoolingHistograms(2))
    //{
    //    Histogram histogram;
    //    for (Descriptor desc : vec)
    //        for (double item : desc)
    //            histogram.Add(item / vec.Count());

    //    histograms.Add(histogram);
    //}
    //printf("%d %d\n", histograms.Count(), histograms[0].Count());

    ArrayList<ArrayList<size_t>> pass = RandomSplit(nImage, nFold);
    for (int i = 0; i < nFold; i++)
    {
        printf("Begin Fold %d...\n", i + 1);
        ArrayList<size_t>& pickUpIndexes = pass[i];
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
        ArrayList<Histogram> trainingHistograms = Divide(histograms, pickUpIndexes).Item2();
        ArrayList<Histogram> evaluationHistograms = Divide(histograms, pickUpIndexes).Item1();

        double precision = KNN<Histogram>().
            Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).first;

        printf("Fold %d Accuracy: %f\n", i + 1, precision);
    }
}