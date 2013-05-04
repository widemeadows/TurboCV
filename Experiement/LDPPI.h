#pragma once

#include "../System/System.h"
#include "../System.Image/System.Image.h"
#include "../System.Image/Eigen.h"
#include "../System.ML/System.ML.h"
#include <cassert>
#include <unordered_map>
#include <cv.h>
using namespace TurboCV::System;
using namespace TurboCV::System::Image;
using namespace TurboCV::System::ML;

template<typename T>
class LDPPI
{
public:
    LDPPI(const ArrayList<T>& data, const ArrayList<int>& labels)
    {
        assert(data.Count() == labels.Count() && data.Count() > 0);

        int categoryNum = 0;
        for (int i = 0; i < data.Count(); i++)
        {
            std::unordered_map<int, int>::iterator itr = _mapping.find(labels[i]);
            if (itr == _mapping.end())
                _mapping.insert(std::make_pair(labels[i], categoryNum++));
        }

		cv::Mat samples;
        ArrayList<cv::Mat> categories(categoryNum);
        for (int i = 0; i < data.Count(); i++)
        {
            cv::Mat row(1, data[i].Count(), CV_64F);
            for (int j = 0; j < data[i].Count(); j++)
                row.at<double>(0, j) = data[i][j];

            categories[_mapping[labels[i]]].push_back(row);
			samples.push_back(row);
        }

		cv::Mat means;
        for (int i = 0; i < categoryNum; i++)
        {
            cv::Mat mean = cv::Mat::zeros(1, categories[i].cols, CV_64F);

            for (int j = 0; j < categories[i].rows; j++)
                for (int k = 0; k < categories[i].cols; k++)
                    mean.at<double>(0, k) += categories[i].at<double>(j, k);

            for (int j = 0; j < categories[i].cols; j++)
                mean.at<double>(0, j) /= categories[i].rows;

            means.push_back(mean);
        }

        _similarity = GetSimilarityMatrix(data);
		
		cv::Mat tmp = cv::Mat::zeros(_similarity.size(), CV_64F);
		for (int i = 0; i < _similarity.rows; i++)
			for (int j = 0; j < _similarity.cols; j++)
				tmp.at<double>(i, i) += _similarity.at<double>(i, j);

		cv::Mat Ln = tmp - _similarity;

		ArrayList<int> normalizedLabels(labels.Count());
		for (int i = 0; i < labels.Count(); i++)
			normalizedLabels[i] = _mapping[labels[i]];
		_confusion = GetConfusionMatrix(data, normalizedLabels);

		tmp = cv::Mat::zeros(_confusion.size(), CV_64F);
		for (int i = 0; i < _confusion.rows; i++)
			for (int j = 0; j < _confusion.cols; j++)
				tmp.at<double>(i, i) += _confusion.at<double>(i, j);

		cv::Mat Ls = tmp - _confusion;
		
		cv::Mat left = samples.t() * Ln * samples;
		cv::Mat right = means.t() * Ls * means;

		cv::Mat eigenValues, eigenVectors;
		eigen(left, right, eigenValues, eigenVectors);

		printf("1\n");
    }

private:
    cv::Mat GetSimilarityMatrix(
        const ArrayList<T>& samples, 
        double sigma = 0.4, 
        double (*GetDistance)(const T&, const T&) = Math::NormOneDistance, 
        int K = 20)
    {
        ArrayList<ArrayList<int>> topIndexes(samples.Count());

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
                topIndexes[i].Add(distanceAndIndexes[j].Item2());
        }

        cv::Mat similarityMatrix(samples.Count(), samples.Count(), CV_64F);
        #pragma omp parallel for
        for (int i = 0; i < samples.Count(); i++)
        {
            for (int j = i; j < samples.Count(); j++)
            {
                double similarity = 0;

                if (topIndexes[i].Contains(j) && topIndexes[j].Contains(i))
                    similarity = Math::Gauss(distanceMatrix.at<double>(i, j), sigma);

                similarityMatrix.at<double>(j, i) = similarityMatrix.at<double>(i, j) = similarity;
            }
        }

        return similarityMatrix;
    }

	// Attention: labels should be normalized to [0,...,C-1], where C is the category number.
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

		std::unordered_map<int, int> sampleNumPerCategory;
		for (int i = 0; i < labels.Count(); i++)
		{
			sampleNumPerCategory[labels[i]]++;
		}

        cv::Mat confusionMatrix(sampleNumPerCategory.size(), sampleNumPerCategory.size(), CV_64F);
		for (int i = 0; i < labels.Count(); i++)
			confusionMatrix.at<double>(labels[i], predictLabels[i])++;
        
		for (int i = 0; i < confusionMatrix.rows; i++)
			for (int j = 0; j < confusionMatrix.cols; j++)
				confusionMatrix.at<double>(i, j) /= sampleNumPerCategory[i];

        return confusionMatrix;
    }

    std::unordered_map<int, int> _mapping;
	cv::Mat _similarity, _confusion;
};