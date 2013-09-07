#include "EdgeMatching.h"
#include <cv.h>
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        ArrayList<PointList> Hitmap::GetChannels(const Mat& sketchImage)
        {
            return GetEdgelChannels(sketchImage, orientNum);
        }

        ArrayList<Mat> Hitmap::GetTransforms(const Size& size, const ArrayList<PointList> channels)
        {
            ArrayList<Mat> result(channels.Count());

            for (size_t i = 0; i < channels.Count(); i++)
            {
                Mat dt(size, CV_8U);
                dt = Scalar::all(0);

                for (size_t j = 0; j < channels[i].Count(); j++)
                {
                    int left = (int)floor(channels[i][j].x - maxDistance),
                        right = (int)ceil(channels[i][j].x + maxDistance),
                        top = (int)floor(channels[i][j].y - maxDistance),
                        bottom = (int)ceil(channels[i][j].y + maxDistance);
                    left = left < 0 ? 0 : left;
                    right = right > dt.cols ? dt.cols : right;
                    top = top < 0 ? 0 : top;
                    bottom = bottom > dt.rows ? dt.rows : bottom;

                    for (int m = top; m < bottom; m++)
                    {
                        for (int n = left; n < right; n++)
                        {
                            double distance = sqrt((m - channels[i][j].y) * (m - channels[i][j].y) + 
                                (n - channels[i][j].x) * (n - channels[i][j].x));

                            if (distance <= maxDistance)
                                dt.at<uchar>(m, n) = 1;
                        }
                    }
                }

                result[i] = dt;
            }

            return result;
        }

        double Hitmap::GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v)
        {
            assert(u.Count() == v.Count());
            int orientNum = u.Count(), uPointNum = 0;
            double uToV = 0;

            for (int i = 0; i < orientNum; i++)
            {
                const PointList& uPoints = u[i];
                const Mat& vMat = v[i];

                for (size_t i = 0; i < uPoints.Count(); i++)
                    uToV += vMat.at<uchar>(uPoints[i].y, uPoints[i].x);

                uPointNum += uPoints.Count();
            }

            if (uPointNum == 0)
                return 1;
            else
                return 1 - uToV / uPointNum;
        }

        double Hitmap::GetTwoWayDistance(double uToV, double vToU)
        {
            return 1 - sqrt((1 - uToV) * (1 - vToU));
        }
    }
}
}