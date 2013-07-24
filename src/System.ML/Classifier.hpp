#pragma once

#include "../System/System.h"
#include <cassert>
#include <utility>
#include <map>
#include <unordered_map>
#include <algorithm>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        //////////////////////////////////////////////////////////////////////////
        // KNN
        //////////////////////////////////////////////////////////////////////////

        const double HARD_VOTING = -1;

        template<typename T>
        class KNN
        {
        public:
            std::pair<double, std::map<std::pair<int, int>, double>> Evaluate(
                const ArrayList<T>& trainingSet,
                const ArrayList<int>& trainingLabels,
                const ArrayList<T>& evaluationSet,
                const ArrayList<int>& evaluationLabels,
                double (*GetDistance)(const T&, const T&) = Math::NormOneDistance,
                double sigma = 0.4,
                int K = 4)
            {
                assert(trainingSet.Count() == trainingLabels.Count());
                assert(evaluationSet.Count() == evaluationSet.Count());

                Train(trainingSet, trainingLabels);
                ArrayList<int> predict = Predict(evaluationSet, GetDistance, sigma, K);

                size_t evaluationNum = evaluationSet.Count(), correctNum = 0;
                std::unordered_map<int, int> evaluationNumPerClass;
                std::map<std::pair<int, int>, double> confusionMatrix;
                for (size_t i = 0; i < evaluationNum; i++)
                {
                    evaluationNumPerClass[evaluationLabels[i]]++;
                    confusionMatrix[std::make_pair(evaluationLabels[i], predict[i])]++;
                    if (predict[i] == evaluationLabels[i])
                        correctNum++;
                }

                std::map<std::pair<int, int>, double>::iterator itr = confusionMatrix.begin();
                while (itr != confusionMatrix.end())
                {
                    itr->second /= evaluationNumPerClass[(itr->first).first];
                    itr++;
                }

                return std::make_pair((double)correctNum / evaluationNum, confusionMatrix);
            }

            void Train(const ArrayList<T>& data, const ArrayList<int>& labels)
            {
                assert(data.Count() == labels.Count() && data.Count() > 0);
                int dataNum = (int)data.Count();

                _labels = labels;
                _data = data;
                _dataNumPerClass.clear();
                for (int i = 0; i < dataNum; i++)
                    _dataNumPerClass[_labels[i]]++;
            }

            ArrayList<int> Predict(
                const ArrayList<T>& samples, 
                double (*GetDistance)(const T&, const T&) = Math::NormOneDistance, 
                double sigma = 0.4, 
                int K = 4)
            {
                int sampleNum = samples.Count();
                ArrayList<int> results(sampleNum);

                #pragma omp parallel for
                for (int i = 0; i < sampleNum; i++)
                {
                    results[i] = predictOneSample(samples[i], GetDistance, sigma, K);
                }

                return results;
            }

        private:
            int predictOneSample(
                const T& sample, 
                double (*GetDistance)(const T&, const T&) = Math::NormOneDistance, 
                double sigma = 0.4, 
                int K = 4)
            {
                size_t dataNum = _data.Count();

                ArrayList<std::pair<double, int>> distances(dataNum);
                for (size_t i = 0; i < dataNum; i++)
                {
                    distances[i] = make_pair(GetDistance(sample, _data[i]), _labels[i]);
                }
                std::partial_sort(distances.begin(), distances.begin() + K, distances.end());

                std::unordered_map<int, double> votes;
                bool softVoting = sigma > 0;
                if (softVoting)
                {
                    for (int i = 0; i < K; i++)
                    {
                        double& distance = distances[i].first;
                        int& label = distances[i].second;
                        votes[label] += Math::Gauss(distance, sigma);
                    }          
                }
                else
                {
                    for (int i = 0; i < K; i++)
                    {
                        int& label = distances[i].second;
                        votes[label]++;
                    } 
                }

                double maxFreq = -1;
                int index = -1;
                for (auto vote : votes)
                {
                    double freq = vote.second / _dataNumPerClass[vote.first];
                    if (freq > maxFreq)
                    {
                        maxFreq = freq;
                        index = vote.first;
                    }
                }

                return index;
            }

            ArrayList<int> _labels;
            ArrayList<T> _data;
            std::unordered_map<int, int> _dataNumPerClass;
        };
    }


    //////////////////////////////////////////////////////////////////////////
    // MQDF
    //////////////////////////////////////////////////////////////////////////

    template<typename T>
    class MQDF
    {
    public:
        std::pair<double, std::map<std::pair<int, int>, double>> Evaluate(
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
            std::unordered_map<int, int> evaluationNumPerClass;
            std::map<std::pair<int, int>, double> confusionMatrix;
            for (size_t i = 0; i < evaluationNum; i++)
            {
                evaluationNumPerClass[evaluationLabels[i]]++;
                confusionMatrix[std::make_pair(evaluationLabels[i], predict[i])]++;
                if (predict[i] == evaluationLabels[i])
                    correctNum++;
            }

            std::map<std::pair<int, int>, double>::iterator itr = confusionMatrix.begin();
            while (itr != confusionMatrix.end())
            {
                itr->second /= evaluationNumPerClass[(itr->first).first];
                itr++;
            }

            return std::make_pair((double)correctNum / evaluationNum, confusionMatrix);
        }

        void Train(const ArrayList<T>& data, const ArrayList<int>& labels)
        {
            assert(data.Count() == labels.Count() && data.Count() > 0);

            int dataNum = data.Count();
            _mapping.clear();
            _invCovariance.Clear();
            _detCovariance.Clear();
            _means.Clear();
            _weights.Clear();

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
                _invCovariance.Add(cv::Mat());
                _detCovariance.Add(0);
                _means.Add(cv::Mat());
                _weights.Add(0);
            }

            printf("Train...\n");
            #pragma omp parallel for
            for (int index = 0; index < categoryNum; index++)
            {
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

        ArrayList<int> Predict(const ArrayList<T>& samples)
        {
            printf("Predict...\n");

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

            ArrayList<Group<double, int>> distanceAndLabels;

            for (auto item : _mapping)
            {
                int index = item.second;

                cv::Mat dif;
                ((cv::Mat)(x - _means[index])).convertTo(dif, CV_32F);

                double distance = ((cv::Mat)(dif * _invCovariance[index] * dif.t())).
                    at<float>(0, 0) - 2 * _weights[index] /*+ _detCovariance[index]*/;

                distanceAndLabels.Add(CreateGroup(distance, item.first));
            }

            return Math::Min(distanceAndLabels).Item2();
        }

        std::unordered_map<int, int> _mapping;
        ArrayList<cv::Mat> _invCovariance;
        ArrayList<double> _detCovariance;
        ArrayList<cv::Mat> _means;
        ArrayList<double> _weights;
    };
}
}