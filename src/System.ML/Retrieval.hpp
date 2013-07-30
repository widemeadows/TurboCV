#include "../System/System.h"
#include <algorithm>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        class MAP
        {
        public:
            template<typename T>
            Group<ArrayList<double>, ArrayList<ArrayList<int>>> Evaluate(
                const ArrayList<T>& trainingSet,
                const ArrayList<int>& trainingLabels,
                const ArrayList<T>& evaluationSet,
                const ArrayList<int>& evaluationLabels,
                double (*GetDistance)(const T&, const T&) = Math::NormOneDistance)
            {
                int nQuery = evaluationLabels.Count(),
                    nDataset = trainingLabels.Count();
                ArrayList<ArrayList<double>> distanceMatrix(nQuery);

                #pragma omp parallel for
                for (int i = 0; i < nQuery; i++)
                {
                    ArrayList<double> distances(nDataset);
                    for (int j = 0; j < nDataset; j++)
                        distances[j] = GetDistance(evaluationSet[i], trainingSet[j]);

                    distanceMatrix[i] = distances;
                }

                return Evaluate(distanceMatrix, trainingLabels, evaluationLabels);
            }

            Group<ArrayList<double>, ArrayList<ArrayList<int>>> Evaluate(
                const ArrayList<ArrayList<double>>& distanceMatrix, 
                const ArrayList<int>& trainingLabels,
                const ArrayList<int>& evaluationLabels)
            {
                int nQuery = evaluationLabels.Count(),
                    nDataset = trainingLabels.Count();
                assert(distanceMatrix.Count() == nQuery);
                assert(distanceMatrix[0].Count() == nDataset);

                ArrayList<ArrayList<double>> apPerQuery(nQuery);
                ArrayList<ArrayList<int>> correctIdxPerQuery(nQuery);

                #pragma omp parallel for
                for (int i = 0; i < nQuery; i++)
                {
                    int nCorrect = 0;
                    double numberator = 0;
                    ArrayList<double> ap(nDataset);
                    ArrayList<int> sortedLabels = SortDistances(distanceMatrix[i], trainingLabels).Item2();
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
                const ArrayList<double>& distances, 
                const ArrayList<int>& labels)
            {
                assert(distances.Count() == labels.Count());

                ArrayList<Group<double, int>> distAndLabels;
                for (int i = 0; i < distances.Count(); i++)
                    distAndLabels.Add(CreateGroup(distances[i], labels[i]));
                
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