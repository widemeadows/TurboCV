#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include "../System.XML/System.XML.h"
#include "CrossValidation.hpp"
#include "Util.h"
#include <cv.h>
#include <highgui.h>
#include <mat.h>
#include <H5Cpp.h>
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

cv::Mat mayaPreprocess(const cv::Mat& image)
{
    int leftPadding = 0, rightPadding = 0, topPadding = 0, bottomPadding = 0;
    if (image.rows < image.cols)
    {
        topPadding = (image.cols - image.rows) / 2;
        bottomPadding = 512 - image.rows - topPadding;
    }
    else
    {
        leftPadding = (image.rows - image.cols) / 2;
        rightPadding = 512 - image.cols - leftPadding;
    }

    cv::Mat squareImage;
    copyMakeBorder(image, squareImage, topPadding, bottomPadding, leftPadding, rightPadding, 
        BORDER_CONSTANT, Scalar(0, 0, 0, 0));
    assert(squareImage.rows == 512 && squareImage.cols == 512);

    cv::Mat scaledImage;
    resize(squareImage, scaledImage, Size(256, 256), 0.0, 0.0, CV_INTER_AREA);

    cv::Mat finalImage;
    threshold(scaledImage, finalImage, 54, 255, CV_THRESH_BINARY);

    return finalImage;
}

cv::Mat oraclePreprocess(const Mat& image)
{
    Mat boundingBox = GetBoundingBox(reverse(image));

    Mat squareImage;
    int leftPadding = 0, topPadding = 0;
    if (boundingBox.rows < boundingBox.cols)
        topPadding = (boundingBox.cols - boundingBox.rows) / 2;
    else
        leftPadding = (boundingBox.rows - boundingBox.cols) / 2;
    copyMakeBorder(boundingBox, squareImage, topPadding, topPadding, 
        leftPadding, leftPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

    Mat scaledImage;
    resize(squareImage, scaledImage, Size(228, 228));

    Mat paddedImage;
    copyMakeBorder(scaledImage, paddedImage, 14, 14, 14, 14, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
    assert(paddedImage.rows == 256 && paddedImage.cols == 256);

    Mat binaryImage;
    threshold(paddedImage, binaryImage, 200, 255, CV_THRESH_BINARY);

    Mat finalImage;
    thin(binaryImage, finalImage);

    return finalImage;
}

void Batch(const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    LocalFeatureCrossValidation<HOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RHOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<SHOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<LogSHOG>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<HOOSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RHOOSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<SC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<PSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RPSC>(datasetPath, preprocess);
    printf("\n");

    LocalFeatureCrossValidation<RGabor>(datasetPath, preprocess);
    printf("\n");

    GlobalFeatureCrossValidation<GHOG>(datasetPath, preprocess);
    printf("\n");

    GlobalFeatureCrossValidation<GIST>(datasetPath, preprocess);
    printf("\n");

    EdgeMatchCrossValidation<CM>(datasetPath, preprocess);
    printf("\n");

    EdgeMatchCrossValidation<OCM>(datasetPath, preprocess);
    printf("\n");

    EdgeMatchCrossValidation<Hitmap>(datasetPath, preprocess);
    printf("\n");
}

void Choose(const TString& algoName, const TString& datasetPath, cv::Mat (*preprocess)(const cv::Mat&))
{
    if (algoName == TString("hog"))
    {
        LocalFeatureCrossValidation<HOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rhog"))
    {
        LocalFeatureCrossValidation<RHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("shog"))
    {
        LocalFeatureCrossValidation<SHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("logshog"))
    {
        LocalFeatureCrossValidation<LogSHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("hoosc"))
    {
        LocalFeatureCrossValidation<HOOSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rhoosc"))
    {
        LocalFeatureCrossValidation<RHOOSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("sc"))
    {
        LocalFeatureCrossValidation<SC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("psc"))
    {
        LocalFeatureCrossValidation<PSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rsc"))
    {
        LocalFeatureCrossValidation<RSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rpsc"))
    {
        LocalFeatureCrossValidation<RPSC>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("rgabor"))
    {
        LocalFeatureCrossValidation<RGabor>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("ghog"))
    {
        GlobalFeatureCrossValidation<GHOG>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("gist"))
    {
        GlobalFeatureCrossValidation<GIST>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("cm"))
    {
        EdgeMatchCrossValidation<CM>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("ocm"))
    {
        EdgeMatchCrossValidation<OCM>(datasetPath, preprocess);
        printf("\n");
    }
    else if (algoName == TString("hitmap"))
    {
        EdgeMatchCrossValidation<Hitmap>(datasetPath, preprocess);
        printf("\n");
    }
}

Group<ArrayList<Word_f>, ArrayList<Histogram>, ArrayList<int>> LoadLocalFeatureData(const TString& fileName)
{
    FILE* file = fopen(fileName, "r");
    int nRow, nCol;

    fscanf(file, "%d %d", &nRow, &nCol);

    ArrayList<Word_f> words(nRow);

    for (int i = 0; i < nRow; i++)
    {
        Word_f word(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &word[j]);

        words[i] = word;
    }

    fscanf(file, "%d %d", &nRow, &nCol);

    ArrayList<int> labels(nRow);
    ArrayList<Histogram> histograms(nRow);

    for (int i = 0; i < nRow; i++)
    {
        fscanf(file, "%d", &labels[i]);

        Histogram histogram(nCol);
        for (int j = 0; j < nCol; j++)
            fscanf(file, "%f", &histogram[j]);

        histograms[i] = histogram;
    }

    fclose(file);

    return CreateGroup(words, histograms, labels);
}

template<typename T>
void MatlabSave(const TString& fileName, const Mat& mat, const TString& variable)
{
    int M = mat.rows;
    int N = mat.cols;
    double *data = (double*)mxCalloc(M * N, sizeof(double));

    // matlab中矩阵是按列存储的
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        data[j * M + i] = mat.at<T>(i, j);

    MATFile *pmatFile = matOpen(fileName, "w");
    mxArray *pMxArray = mxCreateDoubleMatrix(M, N, mxREAL);
    mxSetData(pMxArray, data);
    matPutVariable(pmatFile, variable, pMxArray);

    matClose(pmatFile);
}

template<typename T>
Mat MatlabLoad(const TString& fileName, const TString& variable)
{
    MATFile *pmatFile = matOpen(fileName, "r");
    mxArray *pMxArray = matGetVariable(pmatFile, variable);
    T *data = (T*)mxGetData(pMxArray);
    int M = mxGetM(pMxArray);
    int N = mxGetN(pMxArray);

    // matlab中矩阵是按列存储的
    Mat mat(M, N, CV_64F);
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        mat.at<T>(i, j) = data[j * M + i];

    matClose(pmatFile);

    return mat;
}

void HDF5Save(const char* fileName, const Mat& mat, const char* variable)
{
    H5::H5File file(fileName, H5F_ACC_TRUNC);

    hsize_t dims[2] = { mat.rows, mat.cols };
    H5::DataSpace dataspace(2, dims);

    H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE);  // little endian

    H5::DataSet dataset = file.createDataSet(variable, datatype, dataspace);

    dataset.write(mat.ptr<float>(), H5::PredType::NATIVE_FLOAT);
}

int main(int argc, char* argv[])
{
    //LocalFeatureCrossValidation<RHOG>("subset", sketchPreprocess);
    //return 0;

    //LocalFeatureCrossValidation<HOOSC>("sketches", sketchPreprocess);
    //LocalFeatureCrossValidation<HOOSC>("oracles", oraclePreprocess);
    //return 0;

    auto dataset = LoadDataset("sketches");
    ArrayList<TString> paths = dataset.Item1();
    ArrayList<int> labels = dataset.Item2();
    auto evaIdxes = SplitDatasetRandomly(labels, 3);

    //int nImage = paths.Count();
    //int nDescPerImage = 28 * 28;
    //int nDesc = nImage * nDescPerImage;
    //int dim = 4 * 4 * 9;
    //
    //Mat mat(dim, nDesc, CV_64F);

    //#pragma omp parallel for
    //for (int i = 0; i < nImage; i++)
    //{
    //    Mat image = imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE);
    //    image = sketchPreprocess(image);
    //    auto feature = RGabor()(image);

    //    for (int j = 0; j < dim; j++)
    //    for (int k = 0; k < feature.Count(); k++)
    //        mat.at<double>(j, i * nDescPerImage + k) = feature[k][j];
    //}

    int nImage = paths.Count();
    int dim = 4 * 4 * 9 * 28 * 28;

    Mat mat(nImage, dim, CV_32F);
    Mat y(nImage, 1, CV_32S);

    #pragma omp parallel for
    for (int i = 0; i < nImage; i++)
    {
        Mat image = imread(paths[i], CV_LOAD_IMAGE_GRAYSCALE);
        image = sketchPreprocess(image);
        auto feature = RHOG()(image);

        for (int j = 0; j < feature.Count(); j++)
        for (int k = 0; k < feature[j].Count(); k++)
            mat.at<float>(i, j * feature[j].Count() + k) = feature[j][k];

        y.at<int>(i, 0) = labels[i] - 1;
    }

    H5::H5File file("raw.mat", H5F_ACC_TRUNC);

    hsize_t dims[2] = { mat.rows, mat.cols };
    H5::DataSpace dataspace(2, dims);

    H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE);  // little endian

    H5::DataSet d = file.createDataSet("x", datatype, dataspace);
    d.write(mat.ptr<float>(), H5::PredType::NATIVE_FLOAT);

    dims[0] = y.rows;
    dims[1] = y.cols;
    dataspace = H5::DataSpace(2, dims);

    H5::IntType datatype2(H5::PredType::NATIVE_INT32);
    datatype2.setOrder(H5T_ORDER_LE);

    d = file.createDataSet("y", datatype2, dataspace);
    d.write(y.ptr<int>(), H5::PredType::NATIVE_INT32);
    //MatlabSave<double>("raw.mat", mat, "raw");
    return 0;

    Mat hist = MatlabLoad<double>("hist.mat", "hist");
    ArrayList<Histogram> histograms(hist.cols);
    for (int i = 0; i < hist.cols; i++)
    {
        Histogram histogram(hist.rows);
        for (int j = 0; j < hist.rows; j++)
            histogram[j] = hist.at<double>(j, i);

        histograms[i] = histogram;
    }

    for (int i = 0; i < evaIdxes.Count(); i++)
    {
        printf("\nBegin Fold %d...\n", i + 1);
        const ArrayList<size_t>& pickUpIndexes = evaIdxes[i];
        ArrayList<int> trainingLabels = Divide(labels, pickUpIndexes).Item2();
        ArrayList<int> evaluationLabels = Divide(labels, pickUpIndexes).Item1();
        ArrayList<Histogram> trainingHistograms = Divide(histograms, pickUpIndexes).Item2();
        ArrayList<Histogram> evaluationHistograms = Divide(histograms, pickUpIndexes).Item1();

        auto accuracy = KNN<Histogram>().
            Evaluate(trainingHistograms, trainingLabels, evaluationHistograms, evaluationLabels).Item1();

        printf("Fold %d Accuracy: %f\n", i + 1, accuracy);
    }
}