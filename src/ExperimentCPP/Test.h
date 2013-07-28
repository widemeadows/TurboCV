#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
#include <cv.h>
#include <highgui.h>
#include <random>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;
using namespace cv;
using namespace std;

class TestLevel1: public LocalFeature
{
public:
    TestLevel1() : blockSize(12), descNum(30) {}

    virtual LocalFeatureVec operator()(const cv::Mat& image)
    {
        LocalFeatureVec feature;
        int rowUpperBound = image.rows - blockSize,
            colUpperBound = image.cols - blockSize;
        default_random_engine rowGenerator(100), colGenerator(101);
        uniform_int_distribution<int> rowDistribution(0, rowUpperBound),
            colDistribution(0, colUpperBound);

        for (int i = 0; i < descNum; i++)
        {
            Descriptor desc;
            double mean, std;

            do
            {
                desc = GetBlock(image, rowDistribution(rowGenerator), colDistribution(colGenerator));
            } while (desc.Count() == 0);

            mean = Math::Mean(desc);
            std = Math::StandardDeviation(desc);

            for (int i = desc.Count() - 1; i >= 0; i--)
                desc[i] = (desc[i] - mean) / std;

            feature.Add(desc);
        }

        return feature;
    }

    virtual TString GetName() const
    {
        return "level1";
    }

private:
    Descriptor GetBlock(const Mat& image, int top, int left)
    {
        Mat patch(image, Range(top, top + blockSize), Range(left, left + blockSize));
        double result = sum(patch)[0];

        if (result == 0)
            return Descriptor();

        Mat boundingBox = GetBoundingBox(patch);

        Mat squareImage;
        int widthPadding = 0, heightPadding = 0;
        if (boundingBox.rows < boundingBox.cols)
            heightPadding = (boundingBox.cols - boundingBox.rows) / 2;
        else
            widthPadding = (boundingBox.rows - boundingBox.cols) / 2;
        copyMakeBorder(boundingBox, squareImage, heightPadding, heightPadding, 
            widthPadding, widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));

        Mat paddedImage;
        heightPadding = (blockSize - squareImage.rows) / 2;
        widthPadding = (blockSize - squareImage.cols) / 2; 
        copyMakeBorder(squareImage, paddedImage, heightPadding, blockSize - squareImage.rows - heightPadding,
            widthPadding, blockSize - squareImage.cols - widthPadding, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
        assert(paddedImage.rows == blockSize && paddedImage.cols == blockSize);

        Descriptor desc;
        for (int i = 0; i < patch.rows; i++)
            for (int j = 0; j < patch.cols; j++)
                desc.Add(patch.at<uchar>(i, j));

        return desc;
    }

    int blockSize, descNum;
};

inline void test(const TString& datasetPath)
{
    DirectoryInfo imageSetInfo(datasetPath);
    ArrayList<TString> classInfos = imageSetInfo.GetDirectories();
    sort(classInfos.begin(), classInfos.end());

    ArrayList<TString> imagePaths;
    ArrayList<int> imageLabels;
    for (int i = 0; i < classInfos.Count(); i++)
    {
        ArrayList<TString> fileInfos = DirectoryInfo(classInfos[i]).GetFiles();
        sort(fileInfos.begin(), fileInfos.end());

        for (int j = 0; j < fileInfos.Count(); j++)
        {
            imagePaths.Add(fileInfos[j]);
            imageLabels.Add(i + 1);
        }
    }

    int nImage = imagePaths.Count();
    ArrayList<Descriptor> allDesc;

    //#pragma omp parallel for
    for (int i = 0; i < nImage; i++)
    {
        Mat image = imread(imagePaths[i], CV_LOAD_IMAGE_GRAYSCALE);
        image = reverse(image);
        resize(image, image, Size(32, 32), 0, 0, CV_INTER_AREA);

        LocalFeatureVec feature = TestLevel1()(image);
        for (Descriptor& desc : feature)
            allDesc.Add(desc);
    }

    printf("%d\n", allDesc.Count());
    auto points = PerformTSNE(allDesc);
    printf("%d\n", points.Count());
    FILE* file = fopen("Y.txt", "w");
    for (auto point : points)
    {
        for (auto item : point)
            fprintf(file, "%f ", item);
        fprintf(file, "\n");
    }
    fclose(file);

    file = fopen("labels.txt", "w");
    for (auto point: points)
        fprintf(file, "1\n");
    fclose(file);
}