#include "../System/System.h"
#include <algorithm>
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        template<typename T>
        class MAP
        {
        public:
            Group<ArrayList<double>, ArrayList<ArrayList<int>>> FullCrossValidation(
                const ArrayList<T>& samples,
                const ArrayList<int>& labels,
                double (*GetDistance)(const T&, const T&) = Math::NormOneDistance)
            {
                int nSample = samples.Count();
                cv::Mat distanceMatrix = cv::Mat::zeros(nSample, nSample, CV_64F);

                #pragma omp parallel for
                for (int i = 0; i < nSample; i++)
                    for (int j = 0; j < nSample; j++)
                        distanceMatrix.at<double>(i, j) = GetDistance(samples[i], samples[j]);

                return FullCrossValidation(distanceMatrix, labels);
            }

            Group<ArrayList<double>, ArrayList<ArrayList<int>>> FullCrossValidation(
                const cv::Mat& distanceMatrix,
                const ArrayList<int>& labels)
            {
                int nSample = distanceMatrix.rows;
                ArrayList<ArrayList<double>> apPerQuery(nSample);
                ArrayList<ArrayList<int>> correctIdxPerQuery(nSample);

                #pragma omp parallel for
                for (int i = 0; i < nSample; i++)
                {
                    ArrayList<int> trainingLabels, evaluationLabels;
                    cv::Mat distances = cv::Mat::zeros(1, nSample - 1, CV_64F);
                    int counter = 0;

                    for (int j = 0; j < nSample; j++)
                    {
                        if (j != i)
                        {
                            trainingLabels.Add(labels[j]);

                            distances.at<double>(0, counter) = distanceMatrix.at<double>(i, j);
                            counter++;
                        }
                        else
                        {
                            evaluationLabels.Add(labels[j]);
                        }
                    }

                    auto result = Evaluate(distances, trainingLabels, evaluationLabels);
                    apPerQuery[i] = result.Item1();
                    correctIdxPerQuery[i] = result.Item2()[0];
                }

                ArrayList<double> map(nSample - 1);
                for (int i = 0; i < nSample - 1; i++)
                    for (int j = 0; j < nSample; j++)
                        map[i] += apPerQuery[j][i];
                for (int i = 0; i < nSample - 1; i++)
                    map[i] /= nSample;

                return CreateGroup(map, correctIdxPerQuery);
            }

            Group<ArrayList<double>, ArrayList<ArrayList<int>>> Evaluate(
                const ArrayList<T>& trainingSet,
                const ArrayList<int>& trainingLabels,
                const ArrayList<T>& evaluationSet,
                const ArrayList<int>& evaluationLabels,
                double (*GetDistance)(const T&, const T&) = Math::NormOneDistance)
            {
                int nQuery = evaluationLabels.Count(),
                    nDataset = trainingLabels.Count();
                cv::Mat distanceMatrix = cv::Mat::zeros(nQuery, nDataset, CV_64F);

                #pragma omp parallel for
                for (int i = 0; i < nQuery; i++)
                    for (int j = 0; j < nDataset; j++)
                        distanceMatrix.at<double>(i, j) = GetDistance(evaluationSet[i], trainingSet[j]);

                return Evaluate(distanceMatrix, trainingLabels, evaluationLabels);
            }

            Group<ArrayList<double>, ArrayList<ArrayList<int>>> Evaluate(
                const cv::Mat& distanceMatrix, 
                const ArrayList<int>& trainingLabels,
                const ArrayList<int>& evaluationLabels)
            {
                int nQuery = evaluationLabels.Count(),
                    nDataset = trainingLabels.Count();
                assert(distanceMatrix.rows == nQuery);
                assert(distanceMatrix.cols == nDataset);

                ArrayList<ArrayList<double>> apPerQuery(nQuery);
                ArrayList<ArrayList<int>> correctIdxPerQuery(nQuery);

                #pragma omp parallel for
                for (int i = 0; i < nQuery; i++)
                {
                    int nCorrect = 0;
                    double numberator = 0;
                    ArrayList<double> ap(nDataset);
                    ArrayList<int> sortedLabels = SortDistances(distanceMatrix.row(i), trainingLabels).Item2();
                    assert(sortedLabels.Count() == nDataset);

                    for (int j = 0; j < nDataset; j++)
                    {
                        if (sortedLabels[j] == evaluationLabels[i])
                        {
                            nCorrect++;
                            correctIdxPerQuery[i].Add(j + 1);
                            numberator += nCorrect / (double)(j + 1);
                        }

                        if (nCorrect == 0)
                            ap[j] = 1;
                        else
                            ap[j] = numberator / nCorrect;
                    }

                    apPerQuery[i] = ap;
                }

                ArrayList<double> map(nDataset);
                for (int i = 0; i < nDataset; i++)
                    for (int j = 0; j < nQuery; j++)
                        map[i] += apPerQuery[j][i];
                for (int i = 0; i < nDataset; i++)
                    map[i] /= nQuery;

                return CreateGroup(map, correctIdxPerQuery);
            }

        private:
            Group<ArrayList<double>, ArrayList<int>> SortDistances(
                const cv::Mat& distances, 
                const ArrayList<int>& labels)
            {
                assert(distances.cols == labels.Count());

                ArrayList<Group<double, int>> distAndLabels;
                for (int i = 0; i < distances.cols; i++)
                    distAndLabels.Add(CreateGroup(distances.at<double>(0, i), labels[i]));
                
                sort(distAndLabels.begin(), distAndLabels.end());

                ArrayList<double> sortedDistances(distAndLabels.Count());
                for (int i = 0; i < distAndLabels.Count(); i++)
                    sortedDistances[i] = distAndLabels[i].Item1();

                ArrayList<int> sortedLabels(distAndLabels.Count());
                for (int i = 0; i < distAndLabels.Count(); i++)
                    sortedLabels[i] = distAndLabels[i].Item2();

                return CreateGroup(sortedDistances, sortedLabels);
            }
        };

    }
}
}