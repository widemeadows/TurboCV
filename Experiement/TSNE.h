#include <cassert>
#include <cv.h>
#include "../System/System.h"

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        class TSNE
        {
        public:
            template<typename T>
            void Compute(ArrayList<ArrayList<T>> samples, int dims, double perplexity = 30.0)
            {
                assert(samples.Count() > 0);

                int sampleNum = samples.Count();
                int sampleSize = samples[0].Count();
                int maxIter = 1000;
                double initMomentum = 0.5;
                double finalMomentum = 0.8;
                double eta = 500;
                double minGain = 0.01;

                cv::Mat Y(sampleNum, dims, CV_64F);
                cv::randn(Y, 0, 1);

                cv::Mat dY = cv::Mat::zeros(sampleNum, dims, CV_64F);
                cv::Mat iY = cv::Mat::zeros(sampleNum, dims, CV_64F);
                cv::Mat gains = cv::Mat::ones(sampleNum, dims, CV_64F);


            }

        private:
            void x2p(cv::Mat X, double tolerance = 1e-5, double perplexity = 30.0)
            {
                int n = X.rows, d = X.cols;

            }
        };
    }
}
}