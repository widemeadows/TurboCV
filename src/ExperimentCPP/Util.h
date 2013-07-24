#include "../System/System.h"
#include <map>
using namespace TurboCV::System;

template<typename T, typename Measurement>
std::pair<ArrayList<ArrayList<double>>, ArrayList<ArrayList<bool>>> GetDistanceMatrix(
    const ArrayList<T>& trainingSet,
    const ArrayList<int>& trainingLabels,
    const ArrayList<T>& evaluationSet,
    const ArrayList<int>& evaluationLabels,
    Measurement GetDistance)
{
    assert(trainingSet.Count() == trainingLabels.Count());
    assert(evaluationSet.Count() == evaluationSet.Count());

    ArrayList<ArrayList<double>> distanceMatrix(evaluationSet.Count());
    ArrayList<ArrayList<bool>> relevantMatrix(evaluationSet.Count());

    #pragma omp parallel for
    for (int i = 0; i < evaluationSet.Count(); i++)
    {
        for (size_t j = 0; j < trainingSet.Count(); j++)
        {
            distanceMatrix[i].Add(GetDistance(evaluationSet[i], trainingSet[j]));
            relevantMatrix[i].Add(evaluationLabels[i] == trainingLabels[j]);
        }
    }

    return std::make_pair(distanceMatrix, relevantMatrix);
}

template<typename T>
std::pair<ArrayList<ArrayList<double>>, ArrayList<ArrayList<bool>>> GetDistanceMatrix(
    const ArrayList<T>& trainingSet,
    const ArrayList<int>& trainingLabels,
    const ArrayList<T>& evaluationSet,
    const ArrayList<int>& evaluationLabels)
{
    assert(trainingSet.Count() == trainingLabels.Count());
    assert(evaluationSet.Count() == evaluationSet.Count());

    ArrayList<ArrayList<double>> distanceMatrix(evaluationSet.Count());
    ArrayList<ArrayList<bool>> relevantMatrix(evaluationSet.Count());

    #pragma omp parallel for
    for (int i = 0; i < evaluationSet.Count(); i++)
    {
        for (size_t j = 0; j < trainingSet.Count(); j++)
        {
            distanceMatrix[i].Add(Math::NormOneDistance(evaluationSet[i], trainingSet[j]));
            relevantMatrix[i].Add(evaluationLabels[i] == trainingLabels[j]);
        }
    }

    return std::make_pair(distanceMatrix, relevantMatrix);
}

template<typename T>
Group<ArrayList<double>, ArrayList<double>> ComputeROC(
    const ArrayList<T>& trainingSet,
    const ArrayList<int>& trainingLabels,
    const ArrayList<T>& evaluationSet,
    const ArrayList<int>& evaluationLabels)
{
    int nEva = evaluationSet.Count();
    ArrayList<ArrayList<double>> DRs(nEva), FPRs(nEva);

    pair<ArrayList<ArrayList<double>>, ArrayList<ArrayList<bool>>> matrix =
        GetDistanceMatrix(trainingSet, trainingLabels, evaluationSet, evaluationLabels);
    ArrayList<ArrayList<double>>& distances = matrix.first;
    ArrayList<ArrayList<bool>>& relevants = matrix.second;

    for (int j = 0; j < nEva; j++)
    {
        auto roc = roc(distances[j], relevants[j]);
        DRs[j] = roc.Item1();
        FPRs[j] = roc.Item2();
    }

    return CreateGroup(DRs, FPRs);
}