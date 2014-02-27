#include "../System/System.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        struct GMMFeature
        {
            GMMFeature(
            const ArrayList<double>& a,
            const ArrayList<cv::Mat>& miu,
            const ArrayList<cv::Mat>& sigma) :
            a(a), miu(miu), sigma(sigma) {}

            GMMFeature(const ArrayList<double>& vec)
            {
                int idx = 0;
                int K = vec[idx++];
                int D = vec[idx++];

                for (int i = 0; i < K; i++)
                    a.Add(vec[idx++]);

                for (int i = 0; i < K; i++)
                {
                    cv::Mat tmp(1, D, CV_64F);
                    for (int j = 0; j < D; j++)
                        tmp.at<double>(0, j) = vec[idx++];

                    miu.Add(tmp);
                }

                for (int i = 0; i < K; i++)
                {
                    cv::Mat tmp(D, D, CV_64F);
                    for (int j = 0; j < D; j++)
                    for (int k = 0; k < D; k++)
                        tmp.at<double>(j, k) = vec[idx++];

                    sigma.Add(tmp);
                }
            }

            ArrayList<double> ToArrayList()
            {
                int K = a.Count(), D = miu[0].cols;
                ArrayList<double> vec;

                vec.Add(K);
                vec.Add(D);

                for (int i = 0; i < K; i++)
                    vec.Add(a[i]);

                for (int i = 0; i < K; i++)
                for (int j = 0; j < D; j++)
                    vec.Add(miu[i].at<double>(0, j));

                for (int i = 0; i < K; i++)
                for (int j = 0; j < D; j++)
                for (int k = 0; k < D; k++)
                    vec.Add(sigma[i].at<double>(j, k));

                return vec;
            }

            ArrayList<double> a;
            ArrayList<cv::Mat> miu;
            ArrayList<cv::Mat> sigma;
        };

        double GetDistance(const GMMFeature& u, const GMMFeature& v);

        Group<cv::Mat, GMMFeature> GMM(const cv::Mat& x, int K);
        Group<cv::Mat, GMMFeature> GMM(const cv::Mat& x, int K, const cv::TermCriteria& termCriteria);
    }
}
}