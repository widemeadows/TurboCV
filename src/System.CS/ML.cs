using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TurboCV.System.CS
{
    public delegate double GetDistanceDelegate<T>(T u, T v);

    public class ML
    {
        public static string KNN(
            List<string> sortedTrainLabels,
            List<double> sortedTrainDistances = null, 
            int K = 4)
        {
            Dictionary<string, double> classSize = new Dictionary<string, double>();
            foreach (var label in sortedTrainLabels)
            {
                if (classSize.ContainsKey(label))
                    classSize[label]++;
                else
                    classSize[label] = 0.0;
            }

            string predictLabel = "";

            Dictionary<string, int> dict = new Dictionary<string, int>();
            for (int i = 0; i < K; i++)
            {
                string label = sortedTrainLabels[i];

                if (dict.ContainsKey(label))
                    dict[label]++;
                else
                    dict[label] = 1;
            }

            double max = -1;
            foreach (var item in dict)
            {
                string label = item.Key;
                int num = item.Value;
                double ratio = num / classSize[label];

                if (ratio > max)
                {
                    predictLabel = label;
                    max = ratio;
                }
            }

            return predictLabel;
        }

        public static string KNN<T>(
            List<T> trainFeatures,
            List<string> trainLabels,
            T testFeature,
            GetDistanceDelegate<T> getDistance,
            int K = 4)
        {
            Dictionary<string, double> classSize = new Dictionary<string, double>();
            foreach (var label in trainLabels)
            {
                if (classSize.ContainsKey(label))
                    classSize[label]++;
                else
                    classSize[label] = 0.0;
            }

            string predictLabel = "";
            List<Tuple<double, int>> distAndIndexes = new List<Tuple<double, int>>();
            for (int i = 0; i < trainFeatures.Count; i++)
                distAndIndexes.Add(null);

            Parallel.For(0, trainFeatures.Count, (i) =>
            {
                distAndIndexes[i] = Tuple.Create(getDistance(trainFeatures[i], testFeature), i);
            });
            distAndIndexes.Sort();

            Dictionary<string, int> dict = new Dictionary<string, int>();
            for (int i = 0; i < K; i++)
            {
                string label = trainLabels[distAndIndexes[i].Item2];

                if (dict.ContainsKey(label))
                    dict[label]++;
                else
                    dict[label] = 1;
            }

            double max = -1;
            foreach (var item in dict)
            {
                string label = item.Key;
                int num = item.Value;
                double ratio = num / classSize[label];

                if (ratio > max)
                {
                    predictLabel = label;
                    max = ratio;
                }
            }

            return predictLabel;
        }

        public static List<string> KNN<T>(
            List<T> trainFeatures,
            List<string> trainLabels,
            List<T> testFeatures,
            GetDistanceDelegate<T> getDistance,
            int K = 4)
        {
            Dictionary<string, double> classSize = new Dictionary<string, double>();
            foreach (var label in trainLabels)
            {
                if (classSize.ContainsKey(label))
                    classSize[label]++;
                else
                    classSize[label] = 0.0;
            }

            List<string> predictLabels = new List<string>();
            for (int i = 0; i < testFeatures.Count; i++)
                predictLabels.Add("");

            Parallel.For(0, testFeatures.Count, (i) =>
            {
                List<Tuple<double, int>> distAndIndexes = new List<Tuple<double, int>>();
                for (int j = 0; j < trainLabels.Count; j++)
                {
                    distAndIndexes.Add(Tuple.Create(
                        getDistance(trainFeatures[j], testFeatures[i]), j));
                }

                distAndIndexes.Sort();

                Dictionary<string, int> dict = new Dictionary<string, int>();
                for (int j = 0; j < K; j++)
                {
                    string label = trainLabels[distAndIndexes[j].Item2];

                    if (dict.ContainsKey(label))
                        dict[label]++;
                    else
                        dict[label] = 1;
                }

                double max = -1;
                foreach (var item in dict)
                {
                    string label = item.Key;
                    int num = item.Value;
                    double ratio = num / classSize[label];

                    if (ratio > max)
                    {
                        predictLabels[i] = label;
                        max = ratio;
                    }
                }
            });

            return predictLabels;
        }
    }
}
