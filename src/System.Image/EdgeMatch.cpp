#include "Recognition.h"
#include <numeric>
#include <cv.h>
using namespace std;
using namespace cv;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // CM
        //////////////////////////////////////////////////////////////////////////

        ArrayList<PointList> CM::GetChannels(const Mat& sketchImage)
        {
            return GetEdgelChannels(sketchImage, 1);
        }

        ArrayList<Mat> CM::GetTransforms(const Size& size, const ArrayList<PointList> channels)
        {
            ArrayList<Mat> result(channels.Count());
            Mat dt(size, CV_64F);

            for (int i = 0; i < dt.rows; i++)
                for (int j = 0; j < dt.cols; j++)
                    dt.at<double>(i, j) = maxDistance * maxDistance;

            for (size_t i = 0; i < channels[0].Count(); i++)
            {
                int left = (int)floor(channels[0][i].x - maxDistance),
                    right = (int)ceil(channels[0][i].x + maxDistance),
                    top = (int)floor(channels[0][i].y - maxDistance),
                    bottom = (int)ceil(channels[0][i].y + maxDistance);
                left = left < 0 ? 0 : left;
                right = right > dt.cols ? dt.cols : right;
                top = top < 0 ? 0 : top;
                bottom = bottom > dt.rows ? dt.rows : bottom;

                for (int m = top; m < bottom; m++)
                {
                    for (int n = left; n < right; n++)
                    {
                        double distance = (m - channels[0][i].y) * (m - channels[0][i].y) + 
                            (n - channels[0][i].x) * (n - channels[0][i].x);

                        dt.at<double>(m, n) = min(distance, dt.at<double>(m, n));
                    }
                }
            }

            result[0] = dt;
            return result;
        }

        double CM::GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v)
        {
            assert(u.Count() == v.Count());
            int uPointNum = u[0].Count();
            double uToV = 0;

            for (size_t i = 0; i < uPointNum; i++)
                uToV += v[0].at<double>(u[0][i].y, u[0][i].x);

            if (uPointNum == 0)
                return numeric_limits<double>::max();
            else
                return uToV / uPointNum;
        }

        double CM::GetTwoWayDistance(double uToV, double vToU)
        {
            return (uToV + vToU) / 2.0;
        }


        //////////////////////////////////////////////////////////////////////////
        // OCM
        //////////////////////////////////////////////////////////////////////////

        ArrayList<PointList> OCM::GetChannels(const Mat& sketchImage)
        {
            return GetEdgelChannels(sketchImage, orientNum);
        }

        ArrayList<Mat> OCM::GetTransforms(const Size& size, const ArrayList<PointList> channels)
        {
            ArrayList<Mat> result(channels.Count());

            for (size_t i = 0; i < channels.Count(); i++)
            {
                Mat dt(size, CV_32F);

                for (int j = 0; j < dt.rows; j++)
                    for (int k = 0; k < dt.cols; k++)
                        dt.at<float>(j, k) = maxDistance * maxDistance;

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
                            double distance = (m - channels[i][j].y) * (m - channels[i][j].y) + 
                                (n - channels[i][j].x) * (n - channels[i][j].x);

                            dt.at<float>(m, n) = min(distance, (double)dt.at<float>(m, n));
                        }
                    }
                }

                result[i] = dt;
            }

            return result;
        }

        double OCM::GetOneWayDistance(const ArrayList<PointList>& u, const ArrayList<cv::Mat>& v)
        {
            assert(u.Count() == v.Count());
            int orientNum = u.Count(), uPointNum = 0;
            double uToV = 0;

            for (int i = 0; i < orientNum; i++)
            {
                const ArrayList<Point>& uPoints = u[i];
                const Mat& vMat = v[i];

                for (size_t i = 0; i < uPoints.Count(); i++)
                    uToV += vMat.at<float>(uPoints[i].y, uPoints[i].x);

                uPointNum += uPoints.Count();
            }

            if (uPointNum == 0)
                return numeric_limits<double>::max();
            else
                return uToV / uPointNum;
        }

        double OCM::GetTwoWayDistance(double uToV, double vToU)
        {
            return (uToV + vToU) / 2.0;
        }


        //////////////////////////////////////////////////////////////////////////
        // Hitmap
        //////////////////////////////////////////////////////////////////////////

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

                for (int j = 0; j < uPoints.Count(); j++)
                    uToV += vMat.at<uchar>(uPoints[j].y, uPoints[j].x);

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