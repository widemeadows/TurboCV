#include "../System/System.h"
#include <cv.h>

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        Group<cv::Mat, Group<ArrayList<double>, ArrayList<cv::Mat>, ArrayList<cv::Mat>>>
        GMM(const cv::Mat& X, int k);
    }
}
}