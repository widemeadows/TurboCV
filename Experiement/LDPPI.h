#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.ML/System.ML.h"
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;

template<typename T, typename Measurement>
cv::Mat GetSimilarityMatrix(const ArrayList<T>& samples, double sigma = 0.4, 
                            Measurement GetDistance = Math::NormOneDistance, int K = 20)
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

typedef std::map<std::pair<int, int>, double> ConfusionMatrix;

template<typename T>
ConfusionMatrix GetConfusionMatrix(const ArrayList<T>& samples, const ArrayList<int>& labels)
{
    ArrayList<int> predictLabels(samples.Count());

    //#pragma omp parallel for
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

template<typename T>
void LDPPI(const ArrayList<T>& samples, const ArrayList<int>& labels)
{
    ConfusionMatrix confusionMatrix = GetConfusionMatrix(samples, labels);

    for (int i = 1; i <= 2; i++)
    {
        for (int j = 1; j <= 2; j++)
            printf("%f ", confusionMatrix[std::make_pair(i, j)]);
        printf("\n");
    }
}