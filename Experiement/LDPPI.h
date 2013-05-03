#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;

template<typename T>
class LDPPI
{
public:
    void Train(const ArrayList<T>& data, const ArrayList<int>& labels)
    {
        assert(data.Count() == labels.Count() && data.Count() > 0);

        int dataNum = data.Count();
        _mapping.clear();
        _means.Clear();

        int categoryNum = 0;
        for (int i = 0; i < dataNum; i++)
        {
            std::unordered_map<int, int>::iterator itr = _mapping.find(labels[i]);
            if (itr == _mapping.end())
                _mapping.insert(std::make_pair(labels[i], categoryNum++));
        }

        ArrayList<cv::Mat> categories(categoryNum);
        for (int i = 0; i < dataNum; i++)
        {
            cv::Mat row(1, data[i].Count(), CV_64F);
            for (int j = 0; j < data[i].Count(); j++)
                row.at<double>(0, j) = data[i][j];

            categories[_mapping[labels[i]]].push_back(row);
        }

        for (int i = 0; i < categoryNum; i++)
        {
            cv::Mat mean = cv::Mat::zeros(1, categories[i].cols, CV_64F);

            for (int j = 0; j < categories[i].rows; j++)
                for (int k = 0; k < categories[i].cols; k++)
                    mean.at<double>(0, k) += categories[i].at<double>(j, k);

            for (int j = 0; j < categories[i].cols; j++)
                mean.at<double>(0, j) /= categories[i].rows;

            _means.Add(mean);
        }

        #pragma omp parallel for
        for (int index = 0; index < categoryNum; index++)
        {
            printf("Class %d...\n", index + 1);

            const cv::Mat& data = categories[index];
            _weights[index] = (double)data.rows / dataNum;

            cv::Mat covariation, mean;
            cv::calcCovarMatrix(data, covariation, mean, CV_COVAR_ROWS | CV_COVAR_NORMAL);

            cv::Mat eigenValues, eigenVectors;
            cv::eigen(covariation, eigenValues, eigenVectors);

            for (int j = 40; j < eigenValues.rows; j++)
                eigenValues.at<double>(j, 0) = 0.035;

            cv::Mat diag = cv::Mat::zeros(eigenValues.rows, eigenValues.rows, CV_64F);
            for (int j = 0; j < eigenValues.rows; j++)
                diag.at<double>(j, j) = eigenValues.at<double>(j, 0);
            covariation = eigenVectors.t() * diag * eigenVectors;

            ((cv::Mat)covariation.inv()).convertTo(_invCovariance[index], CV_32F);

            double determinant = 0;
            for (int j = 0; j < eigenValues.rows; j++)
                determinant += log(eigenValues.at<double>(j, 0));
            _detCovariance[index] = determinant;

            _means[index] = mean;
        }
    }

private:
    cv::Mat GetSimilarityMatrix(
        const ArrayList<T>& samples, 
        double sigma = 0.4, 
        double (*GetDistance)(const T&, const T&) = Math::NormOneDistance, 
        int K = 20)
    {
        ArrayList<ArrayList<int>> topResults(samples.Count());

        cv::Mat distanceMatrix = Mat::zeros(samples.Count(), samples.Count(), CV_64F);
        #pragma omp parallel for
        for (int i = 0; i < samples.Count(); i++)
        {
            for (int j = i + 1; j < samples.Count(); j++)
            {
                distanceMatrix.at<double>(i, j) = GetDistance(samples[i], samples[j]);
                distanceMatrix.at<double>(j, i) = distanceMatrix.at<double>(i, j);
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < samples.Count(); i++)
        {
            ArrayList<Tuple<double, int>> distanceAndIndexes(samples.Count());
            for (int j = 0; j < samples.Count(); j++)
                distanceAndIndexes.Add(CreateTuple(distanceMatrix.at<double>(i, j), j));

            std::sort(distanceAndIndexes.begin(), distanceAndIndexes.end());

            for (int j = 0; j < K; j++)
                topResults[i].Add(distanceAndIndexes[j].Item2());
        }

        cv::Mat similarityMatrix(samples.Count(), samples.Count(), CV_64F);
        #pragma omp parallel for
        for (int i = 0; i < samples.Count(); i++)
        {
            for (int j = 0; j < samples.Count(); j++)
            {
                double similarity = 0;

                if (topResults[i].Contains(j))
                    similarity = Math::Gauss(distanceMatrix.at<double>(i, j), sigma);

                similarityMatrix.at<double>(i, j) = similarity;
            }
        }

        return similarityMatrix;
    }

    cv::Mat GetConfusionMatrix(const ArrayList<T>& samples, const ArrayList<int>& labels)
    {
        ArrayList<int> predictLabels(samples.Count());

        #pragma omp parallel for
        for (int i = 0; i < samples.Count(); i++)
        {
            ArrayList<T> trainingSet, evaluationSet;
            ArrayList<int> trainingLabels;

            for (int j = 0; j < samples.Count(); j++)
            {
                if (i != j)
                {
                    trainingSet.Add(samples[j]);
                    trainingLabels.Add(labels[j]);
                }
                else
                {
                    evaluationSet.Add(samples[j]);
                }
            }

            KNN<T> knn;
            knn.Train(trainingSet, trainingLabels);
            predictLabels[i] = knn.Predict(evaluationSet)[0];
        }

        ConfusionMatrix confusionMatrix;
        std::unordered_map<int, int> sampleNumPerClass;
        for (int i = 0; i < labels.Count(); i++)
        {
            sampleNumPerClass[labels[i]]++;
            confusionMatrix[std::make_pair(labels[i], predictLabels[i])]++;
        }

        ConfusionMatrix::iterator itr = confusionMatrix.begin();
        while (itr != confusionMatrix.end())
        {
            itr->second /= sampleNumPerClass[(itr->first).first];
            itr++;
        }

        return confusionMatrix;
    }

    std::unordered_map<int, int> _mapping;
    ArrayList<cv::Mat> _means, _similarity, _confusion;
};


//template<typename T>
//void LDPPI(const ArrayList<T>& samples, const ArrayList<int>& labels)
//{
//    ConfusionMatrix confusionMatrix = GetConfusionMatrix(samples, labels);
//
//    
//}