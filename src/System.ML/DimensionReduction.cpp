#include "TSNE.h"
#include "SNE.h"
#include "../System/System.h"
using namespace TurboCV::System;
using namespace TurboCV::System::ML;

ArrayList<ArrayList<double>> normalizeSamples(const ArrayList<ArrayList<double>>& samples)
{
    double maxSum = 0;
    for (int i = 0; i < samples.Count(); i++)
    {
        double sum = 0;
        for (int j = 0; j < samples[i].Count(); j++)
            sum += samples[i][j] * samples[i][j];

        if (sum > maxSum)
            maxSum = sum;
    }

    double root = sqrt(maxSum);
    if (root == 0)
        return samples;
    else
    {
        ArrayList<ArrayList<double>> results(samples.Count());
        for (int i = 0; i < samples.Count(); i++)
        {
            ArrayList<double> tmp(samples[i].Count());
            for (int j = 0; j < samples[i].Count(); j++)
                tmp[j] = samples[i][j] / root;

            results.Add(tmp);
        }

        return results;
    }
}

ArrayList<ArrayList<double>> PerformTSNE(
    const ArrayList<ArrayList<double>>& samples,
    int dimension = 2)
{
    ArrayList<ArrayList<double>> data = normalizeSamples(samples);

    TSNE tsne;
    cv::Mat Y = tsne.Compute(data, 30, dimension);

    ArrayList<ArrayList<double>> results(Y.rows);
    for (int i = 0; i < Y.rows; i++)
    {
        ArrayList<ArrayList<double>> tmp(Y.cols);
        for (int j = 0; j < Y.cols; j++)
            tmp[j] = Y.at<double>(i, j);

        results.Add(tmp);
    }

    return results;
}

ArrayList<ArrayList<double>> PerformSNE(
    const ArrayList<ArrayList<double>>& samples,
    int dimension = 2)
{
    ArrayList<ArrayList<double>> data = normalizeSamples(samples);

    SNE sne;
    cv::Mat Y = sne.Compute(data, 30, dimension);

    ArrayList<ArrayList<double>> results(Y.rows);
    for (int i = 0; i < Y.rows; i++)
    {
        ArrayList<ArrayList<double>> tmp(Y.cols);
        for (int j = 0; j < Y.cols; j++)
            tmp[j] = Y.at<double>(i, j);

        results.Add(tmp);
    }

    return results;
}