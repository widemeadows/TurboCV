#pragma once

#include "../System/System.h"
#include <unordered_map>
#include <algorithm>
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        template<typename T>
        class KNN
        {
        public:
            std::pair<double, std::map<std::pair<int, int>, double>> Evaluate(
                const ArrayList<T>& trainingSet,
                const ArrayList<int>& trainingLabels,
                const ArrayList<T>& evaluationSet,
                const ArrayList<int>& evaluationLabels,
                double (*GetDistance)(const T&, const T&),
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

            std::pair<double, std::map<std::pair<int, int>, double>> Evaluate(
                const ArrayList<T>& trainingSet,
                const ArrayList<int>& trainingLabels,
                const ArrayList<T>& evaluationSet,
                const ArrayList<int>& evaluationLabels,
                double sigma = 0.4,
                int K = 4)
            {
                assert(trainingSet.Count() == trainingLabels.Count());
                assert(evaluationSet.Count() == evaluationSet.Count());

                Train(trainingSet, trainingLabels);
                ArrayList<int> predict = Predict(evaluationSet, sigma, K);

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
                double (*GetDistance)(const T&, const T&), 
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

            ArrayList<int> Predict(const ArrayList<T>& samples, double sigma = 0.4, int K = 4)
            {
                int sampleNum = samples.Count();
                ArrayList<int> results(sampleNum);

                #pragma omp parallel for
                for (int i = 0; i < sampleNum; i++)
                {
                    results[i] = predictOneSample(samples[i], sigma, K);
                }

                return results;
            }

            static const double HARD_VOTING;

        private:
            int predictOneSample(const T& sample, double (*GetDistance)(const T&, const T&), 
                double sigma, int K)
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

            int predictOneSample(const T& sample, double sigma, int K)
            {
                size_t dataNum = _data.Count();
                ArrayList<std::pair<double, int>> distances(dataNum);

                for (size_t i = 0; i < dataNum; i++)
                {
                    distances[i] = make_pair(
                        Math::NormOneDistance(sample, _data[i]), _labels[i]);
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

        template<typename T>
        const double KNN<T>::HARD_VOTING = -1;
    }
}
}