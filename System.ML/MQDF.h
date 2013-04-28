#pragma once

#include "../System/System.h"
#include <cassert>
#include <cmath>
#include <map>
#include <unordered_map>
#include <cv.h>

namespace TurboCV
{
    namespace System
    {
        namespace ML
        {
            template<typename T>
            class MQDF
            {
            public:
                std::pair<double, std::map<int, double>> Evaluate(
                    const ArrayList<T>& trainingSet,
                    const ArrayList<int>& trainingLabels,
                    const ArrayList<T>& evaluationSet,
                    const ArrayList<int>& evaluationLabels)
                {
                    assert(trainingSet.Count() == trainingLabels.Count());
                    assert(evaluationSet.Count() == evaluationSet.Count());

                    Train(trainingSet, trainingLabels);
                    ArrayList<int> predict = Predict(evaluationSet);

                    size_t evaluationNum = evaluationSet.Count(), correctNum = 0;
                    std::unordered_map<int, int> evaluationNumPerClass, correctNumPerClass;
                    for (size_t i = 0; i < evaluationNum; i++)
                    {
                        evaluationNumPerClass[evaluationLabels[i]]++;

                        if (predict[i] == evaluationLabels[i])
                        {
                            correctNum++;
                            correctNumPerClass[evaluationLabels[i]]++;
                        }
                    }

                    std::map<int, double> precisions;
                    for (auto item : _categories)
                    {
                        int label = item.first;
                        precisions[label] = (double)correctNumPerClass[label] / 
                            evaluationNumPerClass[label];
                    }

                    return std::make_pair((double)correctNum / evaluationNum, precisions);
                }

                void Train(const ArrayList<T>& data, const ArrayList<int>& labels)
                {
                    assert(data.Count() == labels.Count() && data.Count() > 0);

                    int dataNum = data.Count();
                    _D = data[0].Count();
                    _categories.clear();
                    _invCovariance.clear();
                    _determine.clear();
                    _means.clear();
                    _weights.clear();

                    for (int i = 0; i < dataNum; i++)
                    {
                        cv::Mat row(1, data[i].Count(), CV_64F);
                        for (int j = 0; j < data[i].Count(); j++)
                            row.at<double>(0, j) = data[i][j];

                        _categories[labels[i]].push_back(row);
                        _weights[labels[i]]++;
                    }

                    std::unordered_map<int, double>::iterator itr = _weights.begin();
                    while (itr != _weights.end())
                    {
                        itr->second /= dataNum;
                        itr++;
                    }

                    for (auto item : _categories)
                    {
                        const int label = item.first;
                        const cv::Mat& data = item.second;

                        cv::Mat covariation, mean;
                        cv::calcCovarMatrix(data, covariation, mean, CV_COVAR_ROWS | CV_COVAR_NORMAL);

                        cv::Mat eigenValues, eigenVectors;
                        cv::eigen(covariation, eigenValues, eigenVectors);

                        double maxValue = -1e10;
                        for (int j = 0; j < eigenValues.rows; j++)
                            maxValue = max(maxValue, eigenValues.at<double>(j, 0));

                        double smallValueMean = 0;
                        int smallValueNum = 0;
                        for (int j = 0; j < eigenValues.rows; j++)
                        {
                            if (eigenValues.at<double>(j, 0) < maxValue / 100)
                            {
                                smallValueMean += eigenValues.at<double>(j, 0);
                                smallValueNum++;
                            }
                        }
                        smallValueMean /= smallValueNum;

                        for (int j = eigenValues.rows - smallValueNum; j < eigenValues.rows; j++)
                            eigenValues.at<double>(j, 0) = smallValueMean;

                        Mat diag = Mat::zeros(eigenValues.rows, eigenValues.rows, CV_64F);
                        for (int j = 0; j < eigenValues.rows; j++)
                            diag.at<double>(j, j) = eigenValues.at<double>(j, 0);

                        covariation = eigenVectors.t() * diag * eigenVectors;
                        _invCovariance[label] = covariation.inv();

                        double determinant = 0;
                        for (int j = 0; j < eigenValues.rows - smallValueNum; j++)
                            determinant += log(eigenValues.at<double>(j, 0));
                        _determine[label] = determinant;

                        _means[label] = mean;
                    }
                }

                ArrayList<int> Predict(const ArrayList<T>& samples)
                {
                    int sampleNum = samples.Count();
                    ArrayList<int> results(sampleNum);

                    #pragma omp parallel for
                    for (int i = 0; i < sampleNum; i++)
                    {
                        results[i] = predictOneSample(samples[i]);
                    }

                    return results;
                }

            private:
                int predictOneSample(const T& sample)
                {
                    cv::Mat x(1, sample.Count(), CV_64F);
                    for (int i = 0; i < sample.Count(); i++)
                        x.at<double>(0, i) = sample[i];

                   ArrayList<Tuple<double, int>> distanceAndLabels;

                    for (auto item : _categories)
                    {
                        int label = item.first;

                        cv::Mat dif = x - _means[label];
                        double distance = ((cv::Mat)(dif * _invCovariance[label] * dif.t())).
                            at<double>(0, 0) - 0.5 * _weights[label]
                            /*+ _determine[label]*/;
                        distanceAndLabels.Add(CreateTuple(distance, label));
                    }

                    return Math::Min(distanceAndLabels).Item2();
                }

                int _D;
                std::unordered_map<int, cv::Mat> _categories, _invCovariance;
                std::unordered_map<int, double> _determine;
                std::unordered_map<int, cv::Mat> _means;
                std::unordered_map<int, double> _weights;
            };
        }
    }
}