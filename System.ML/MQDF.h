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
                    for (auto item : _trainingData)
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
                    _K = _D * 0.5;
                    if (_K > dataNum)
                        _K = dataNum;
                    _trainingData.clear();
                    _eigens.clear();
                    _means.clear();
                    _weights.clear();
                    _constants.clear();

                    for (int i = 0; i < dataNum; i++)
                    {
                        cv::Mat row(1, data[i].Count(), CV_64F);
                        for (int j = 0; j < data[i].Count(); j++)
                            row.at<double>(0, j) = data[i][j];

                        _trainingData[labels[i]].push_back(row);
                        _weights[labels[i]]++;
                    }

                    std::unordered_map<int, double>::iterator itr = _weights.begin();
                    while (itr != _weights.end())
                    {
                        itr->second /= dataNum;
                        itr++;
                    }

                    for (auto item : _trainingData)
                    {
                        const int label = item.first;
                        const cv::Mat& data = item.second;

                        cv::Mat covariation, mean;
                        cv::calcCovarMatrix(data, covariation, mean, CV_COVAR_ROWS | CV_COVAR_NORMAL);

                        cv::Mat eigenValues, eigenVectors;
                        cv::eigen(covariation, eigenValues, eigenVectors);

                        _means[label] = mean;
                        _eigens[label] = CreateTuple(eigenValues, eigenVectors);

                        double constant = 0;
                        const cv::Mat& values = eigenValues;
                        for (int j = _K; j < _D; j++)
                            constant += values.at<double>(j, 0);
                        constant /= (_D - _K);
                        _constants[label] = constant;
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

                    std::map<double, int> distanceAndLabels;

                    for (auto item : _trainingData)
                    {
                        int label = item.first;
                        double factor1 = 0, factor2 = 0;

                        //cv::Mat dif = x - _means[label];
                        //double tmp = cv::norm(dif, cv::NORM_L2);
                        //factor1 = tmp * tmp;
                        //
                        //double constant = _constants[label];
                        //for (int j = 0; j < _K; j++)
                        //{
                        //    const Tuple<cv::Mat, cv::Mat>& item = _eigens[label];
                        //    double eigenValue = item.Item1().at<double>(j, 0);
                        //    cv::Mat eigenVector = item.Item2().row(j);

                        //    double tmp = ((cv::Mat)(eigenVector * dif.t())).at<double>(0, 0);
                        //    factor1 -= (1 - constant / eigenValue) * tmp * tmp;

                        //    factor2 += std::log(eigenValue);
                        //}

                        //factor1 /= constant;
                        //factor2 += (_D - _K) * log(constant);

                        for (int j = 0; j < _D; j++)
                        {
                            const Tuple<cv::Mat, cv::Mat>& item = _eigens[label];
                            double eigenValue = item.Item1().at<double>(j, 0);
                            cv::Mat eigenVector = item.Item2().row(j);

                            if (abs(eigenValue) < 1e-4)
                                continue;

                            double tmp = ((cv::Mat)(eigenVector * (x - _means[label]).t())).at<double>(0, 0);
                            factor1 += tmp * tmp / eigenValue;
                            
                            factor2 += log(eigenValue);
                        }

                        double distance = (factor1 + factor2)/* / _weights[label]*/;
                        distanceAndLabels[distance] = label;
                        printf("%f %d\n", distance, label);
                    }

                    return (distanceAndLabels.begin())->second;
                }

                int _K, _D;
                std::unordered_map<int, cv::Mat> _trainingData;
                std::unordered_map<int, Tuple<cv::Mat, cv::Mat>> _eigens;
                std::unordered_map<int, cv::Mat> _means;
                std::unordered_map<int, double> _constants, _weights;
            };
        }
    }
}