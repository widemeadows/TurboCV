#pragma once

#include "../System/System.h"
#include <cv.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
using namespace cv;
using namespace std;

namespace System
{
    namespace ML
    {
        class KNN
        {
        public:
            static pair<vector<vector<double>>, vector<vector<bool>>> Evaluate(
                const vector<vector<double>>& trainingSet,
                const vector<int>& trainingLabels,
                const vector<vector<double>>& evaluationSet,
                const vector<int>& evaluationLabels)
            {
                assert(trainingSet.size() == trainingLabels.size());
                assert(evaluationSet.size() == evaluationSet.size());

		        vector<vector<double>> distanceMatrix(evaluationSet.size());
                vector<vector<bool>> relevantMatrix(evaluationSet.size());

                #pragma omp parallel for
                for (int i = 0; i < evaluationSet.size(); i++)
                {
                    for (int j = 0; j < trainingSet.size(); j++)
		            {
                        distanceMatrix[i].push_back(Math::NormOneDistance(
                            evaluationSet[i], trainingSet[j]));

                        relevantMatrix[i].push_back(evaluationLabels[i] == trainingLabels[j]);
		            }
                }

                return make_pair(distanceMatrix, relevantMatrix);
            }

            pair<double, map<int, double>> Evaluate(
                const vector<vector<double>>& trainingSet,
                const vector<int>& trainingLabels,
                const vector<vector<double>>& evaluationSet,
                const vector<int>& evaluationLabels,
                int K)
	        {
                assert(trainingSet.size() == trainingLabels.size());
                assert(evaluationSet.size() == evaluationSet.size());

		        Train(trainingSet, trainingLabels);
		        vector<int> predict = Predict(evaluationSet, K);

                int evaluationNum = (int)evaluationSet.size(), correctNum = 0;
                unordered_map<int, int> evaluationNumPerClass, correctNumPerClass;
		        for (int i = 0; i < evaluationNum; i++)
		        {
			        evaluationNumPerClass[evaluationLabels[i]]++;

			        if (predict[i] == evaluationLabels[i])
			        {
				        correctNum++;
				        correctNumPerClass[evaluationLabels[i]]++;
			        }
		        }

		        map<int, double> precisions;
                for (auto item : _dataNumPerClass)
                {
                    int label = item.first;
			        precisions[label] = (double)correctNumPerClass[label] / evaluationNumPerClass[label];
                }

		        return make_pair((double)correctNum / evaluationNum, precisions);
	        }

            void Train(const vector<vector<double>>& data, const vector<int>& labels)
	        {
                assert(data.size() == labels.size() && data.size() > 0);
                int dataNum = (int)data.size();

		        _labels = labels;

		        _data = data;
                for (int i = 1; i < dataNum; i++)
                    assert(_data[i].size() == _data[i - 1].size());

                _dataNumPerClass.clear();
		        for (int i = 0; i < dataNum; i++)
                    _dataNumPerClass[_labels[i]]++;
	        }

            vector<int> Predict(const vector<vector<double>>& samples, int K)
	        {
                int sampleNum = samples.size();
		        vector<int> results(sampleNum);

		        #pragma omp parallel for
		        for (int i = 0; i < sampleNum; i++)
		        {
			        results[i] = predictOneSample(samples[i], K);
		        }

		        return results;
	        }

        private:
            int predictOneSample(const vector<double>& sample, int K)
	        {
                int dataNum = (int)_data.size();
		        vector<pair<double, int>> distances(dataNum);

		        for (int i = 0; i < dataNum; i++)
		        {
			        distances[i] = make_pair(Math::NormOneDistance(sample, _data[i]), _labels[i]);
		        }
		        partial_sort(distances.begin(), distances.begin() + K, distances.end());

		        unordered_map<int, int> votes;
		        for (int i = 0; i < K; i++)
		        {
			        int& label = distances[i].second;
			        votes[label]++;
		        }
		
		        double maxFreq = -1;
		        int index = -1;
		        for (auto vote : votes)
		        {
			        double freq = (double)vote.second / _dataNumPerClass[vote.first];
			        if (freq > maxFreq)
			        {
				        maxFreq = freq;
				        index = vote.first;
			        }
		        }

                return index;
            }

            vector<int> _labels;
            vector<vector<double>> _data;
            unordered_map<int, int> _dataNumPerClass;
        };
    }
}