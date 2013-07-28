#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include "../System.XML/System.XML.h"
#include "Util.h"
#include "Test.h"
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

//template<typename EdgeMatching>
//void EdgeMatchingCrossValidation(const TurboCV::System::TString& imageSetPath, const EdgeMatching& matching,
//                                 bool thinning = false, int fold = 3)
//{
//    srand(1);
//    ArrayList<Group<Mat, int>> images = GetImages(imageSetPath, CV_LOAD_IMAGE_GRAYSCALE);
//    int imageNum = (int)images.Count();
//
//    ArrayList<EdgeMatching::Info> transforms(imageNum);
//    printf("Compute " + matching.GetName() + "...\n");
//    EdgeMatching machine = matching;
//    #pragma omp parallel for private(machine)
//    for (int i = 0; i < imageNum; i++)
//    {
//        transforms[i] = machine.GetFeatureWithPreprocess(images[i].Item1(), thinning);
//        images[i].Item1().release();
//    }
//
//#ifdef SAVE_DISTANCE_MATRIX
//    {
//        Mat distanceMatrix = Mat::zeros(imageNum, imageNum, CV_64F);
//        #pragma omp parallel for
//        for (int i = 0; i < imageNum; i++)
//        {
//            for (int j = 0; j < imageNum; j++)
//            {
//                if (i != j)
//                    distanceMatrix.at<double>(i, j) = EdgeMatching::GetDistance(transforms[i], transforms[j]);
//            }
//        }
//
//        printf("Write To File...\n");
//        TurboCV::System::TString savePath = matching.GetName() + "_" + imageSetPath + "_matrix";
//        FILE* file = fopen(savePath, "w");
//        for (int i = 0; i < imageNum; i++)
//        {
//            fprintf(file, "%d", images[i].Item2());
//            for (int j = 0; j < imageNum; j++)
//            {
//                if (i != j)
//                {
//                    fprintf(file, " %d:%f", images[i].Item2() == images[j].Item2() ? 1 : 0, 
//                        distanceMatrix.at<double>(i, j));
//                }
//            }
//            fprintf(file, "\n");
//        }
//        fclose(file);
//        distanceMatrix.release();
//    }
//#endif
//
//    ArrayList<Group<ArrayList<EdgeMatching::Info>, ArrayList<EdgeMatching::Info>, ArrayList<size_t>>> pass = 
//        RandomSplit(transforms, fold);
//
//#ifdef SAVE_ROC
//    ArrayList<ArrayList<double>> DRs(imageNum), FPRs(imageNum);
//#endif
//
//    ArrayList<double> passResult;
//    for (int i = 0; i < fold; i++)
//    {
//        printf("\nBegin Fold %d...\n", i + 1);
//        ArrayList<EdgeMatching::Info>& evaluationSet = pass[i].Item1();
//        ArrayList<EdgeMatching::Info>& trainingSet = pass[i].Item2();
//        ArrayList<size_t>& pickUpIndexes = pass[i].Item3();
//
//        ArrayList<int> trainingLabels, evaluationLabels;
//        int counter = 0;
//        for (int j = 0; j < imageNum; j++)
//        {
//            if (counter < pickUpIndexes.Count() && j == pickUpIndexes[counter])
//            {
//                evaluationLabels.Add(images[j].Item2());
//                counter++;
//            }
//            else
//                trainingLabels.Add(images[j].Item2());
//        }
//
//        KNN<EdgeMatching::Info> knn;
//        pair<double, map<pair<int, int>, double>> precisions = knn.Evaluate(
//            trainingSet, trainingLabels, evaluationSet, evaluationLabels, 
//            EdgeMatching::GetDistance, KNN<EdgeMatching::Info>::HARD_VOTING);
//
//        passResult.Add(precisions.first);
//        printf("Fold %d Accuracy: %f\n", i + 1, precisions.first);
//
//#ifdef SAVE_ROC
//        pair<ArrayList<ArrayList<double>>, ArrayList<ArrayList<bool>>> matrix = GetDistanceMatrix(
//            trainingSet, trainingLabels, evaluationSet, evaluationLabels, EdgeMatching::GetDistance);
//        ArrayList<ArrayList<double>>& distances = matrix.first;
//        ArrayList<ArrayList<bool>>& relevants = matrix.second;
//        for (int j = 0; j < pickUpIndexes.Count(); j++)
//        {
//            int index = pickUpIndexes[j];
//            auto roc = roc(distances[j], relevants[j]);
//            DRs[index] = roc.Item1();
//            FPRs[index] = roc.Item2();
//        }
//#endif
//    }
//
//#ifdef SAVE_ROC
//    {
//        TurboCV::System::TString savePath = matching.GetName() + "_" + imageSetPath + "_roc";
//        FILE* file = fopen(savePath, "w");
//        for (int i = 0; i < DRs.Count(); i++)
//        {
//            for (int j = 0; j < DRs[i].Count(); j++)
//                fprintf(file, "%f ", DRs[i][j]);
//            fprintf(file, "\n");
//
//            for (int j = 0; j < FPRs[i].Count(); j++)
//                fprintf(file, "%f ", FPRs[i][j]);
//            fprintf(file, "\n");
//        }
//        fclose(file);
//    }
//#endif
//
//    TurboCV::System::TString savePath = matching.GetName() + "_" + imageSetPath + "_knn.out";
//    FILE* file = fopen(savePath, "w");
//    for (int i = 0; i < passResult.Count(); i++)
//        fprintf(file, "Fold %d Accuracy: %f\n", i + 1, passResult[i]);
//    fprintf(file, "Average: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
//        Math::StandardDeviation(passResult));
//    printf("\nAverage: %f, Standard Deviation: %f\n", Math::Mean(passResult), 
//        Math::StandardDeviation(passResult));
//    fclose(file);
//}

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

int main()
{
    test("subset");

    //LocalFeatureCrossValidation(sketchPreprocess, "sketches", HOOSC());


    // Batch();


    //Mat img = imread("00002.png", CV_LOAD_IMAGE_GRAYSCALE);
    //vector<Point> points;
    //goodFeaturesToTrack(img, points, 10, 0.01, 10, noArray(), 3, true, 0.04);

    //Mat tmp;
    //cvtColor(img, tmp, CV_GRAY2BGR);

    //for (auto point : points)
    //{
    //    circle(tmp, point, 1, Scalar(0, 255, 0));
    //}

    //imshow("win", tmp);
    //waitKey(0);

    //system("pause");
}