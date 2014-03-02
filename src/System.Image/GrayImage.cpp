#include "../System/System.h"
#include "Core.h"
#include <cv.h>
#include <highgui.h>
#include <cassert>
#include <unordered_set>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
	namespace Image
	{
        //////////////////////////////////////////////////////////////////////////
        // APIs for Basic Operations
        //////////////////////////////////////////////////////////////////////////

		Mat reverse(Mat image) 
		{
			assert(image.type() == CV_8U);
			Mat result(image.size(), CV_8U);

			for (int i = 0; i < image.rows; i++)
				for (int j = 0; j < image.cols; j++)
					result.at<uchar>(i, j) = 255 - image.at<uchar>(i, j);

			return result;
		}

        Mat FFTShift(const Mat& image)
        {
            int width = image.cols, height = image.rows;
            Mat result(image.rows, image.cols, image.type());

            for (int i = 0; i < height / 2; i++)
            {
                for (int j = 0; j < width / 2; j++)
                    result.at<double>(i, j) = image.at<double>(i + (height + 1) / 2, j + (width + 1) / 2);

                for (int j = 0; j < (width + 1) / 2; j++)
                    result.at<double>(i, j + width / 2) = image.at<double>(i + (height + 1) / 2, j);
            }

            for (int i = 0; i < (height + 1) / 2; i++)
            {
                for (int j = 0; j < width / 2; j++)
                    result.at<double>(i + height / 2, j) = image.at<double>(i, j + (width + 1) / 2);

                for (int j = 0; j < (width + 1) / 2; j++)
                    result.at<double>(i + height / 2, j + width / 2) = image.at<double>(i, j);
            }

            return result;
        }

        void ConvolveFFT(InputArray src, OutputArray dst, int ddepth, InputArray kernel)
        {
            Mat A = src.getMat(), B = kernel.getMat();
            if (ddepth != -1)
            {
                A.convertTo(A, ddepth);
                B.convertTo(B, ddepth);
            }
            CV_Assert(B.rows % 2 != 0 && B.cols % 2 != 0);

            int actualType = ddepth == -1 ? A.type() : ddepth;

            // reallocate the output array if needed
            dst.create(A.rows, A.cols, actualType);
            Mat C = dst.getMat();

            Size imgSize = Size(A.cols + B.cols - 1, A.rows + B.rows - 1);

            // calculate the size of DFT transform
            Size dftSize;
            dftSize.width = getOptimalDFTSize(imgSize.width);
            dftSize.height = getOptimalDFTSize(imgSize.height);

            // allocate temporary buffers and initialize them
            Mat tempA = Mat::zeros(dftSize, actualType);
            Mat roiA(tempA, Rect(B.cols / 2, B.rows / 2, A.cols, A.rows));
            A.copyTo(roiA);
            copyMakeBorder(roiA, tempA(Rect(0, 0, imgSize.width, imgSize.height)), 
                B.rows / 2, B.rows / 2, B.cols / 2, B.cols / 2, 
                BORDER_DEFAULT | BORDER_ISOLATED);

            Mat tempB = Mat::zeros(dftSize, actualType);
            Mat roiB(tempB, Rect(0, 0, B.cols, B.rows));
            B.copyTo(roiB);

            // now transform the padded A & B in-place;
            // use "nonzeroRows" hint for faster processing
            dft(tempA, tempA, 0, imgSize.height);
            dft(tempB, tempB, 0, B.rows);

            // multiply the spectrums;
            // the function handles packed spectrum representations well
            mulSpectrums(tempA, tempB, tempA, 0, true);

            // transform the product back from the frequency domain.
            // Even though all the result rows will be non-zero,
            // you need only the first C.rows of them, and thus you
            // pass nonzeroRows == C.rows
            dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

            // now copy the result back to C.
            tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
        }

        Mat imshow(const Mat& image, bool scale)
        {
            double maximum = 1e-14, minimum = 1e14;
            int type = image.type();
            Mat tmp(image.size(), CV_64F);

            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    double value;

                    if (type == CV_8U)
                        value = image.at<uchar>(i, j);
                    else if (type == CV_64F)
                        value = image.at<double>(i, j);
                    else if (type == CV_32S)
                        value = image.at<int>(i, j);
                    else if (type == CV_8S)
                        value = image.at<char>(i, j);
                    else if (type == CV_32F)
                        value = image.at<float>(i, j);

                    maximum = max(value, maximum);
                    minimum = min(value, minimum);
                    tmp.at<double>(i, j) = value;
                }
            }

            if (maximum > minimum)
            {
                for (int i = 0; i < tmp.rows; i++)
                    for (int j = 0; j < tmp.cols; j++)
                        tmp.at<double>(i, j) = (tmp.at<double>(i, j) - minimum) / (maximum - minimum);
            }

            imshow("OpenCV", tmp);
            return tmp;
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for Gradient
        //////////////////////////////////////////////////////////////////////////
        
        Group<Mat, Mat> GetDiscreteGradient(const Mat& image)
        {
            Mat dxImage(image.size(), CV_64F), dyImage(image.size(), CV_64F);

            auto getPixel = [image](size_t i, size_t j) -> double
            {
                if (image.type() == CV_64F)
                    return image.at<double>(i, j);
                else if (image.type() == CV_32F)
                    return image.at<float>(i, j);
                else if (image.type() == CV_32S)
                    return image.at<int>(i, j);
                else if (image.type() == CV_8U)
                    return image.at<uchar>(i, j);
                else
                    return image.at<char>(i, j);
            };

            for (int i = 0; i < image.rows; i++)
            for (int j = 0; j < image.cols; j++)
            {
                if (i == 0)
                    dyImage.at<double>(i, j) = getPixel(i + 1, j) - getPixel(i, j);
                else if (i == image.rows - 1)
                    dyImage.at<double>(i, j) = getPixel(i, j) - getPixel(i - 1, j);
                else
                    dyImage.at<double>(i, j) = (getPixel(i + 1, j) - getPixel(i - 1, j)) / 2.0;

                if (j == 0)
                    dxImage.at<double>(i, j) = getPixel(i, j + 1) - getPixel(i, j);
                else if (j == image.cols - 1)
                    dxImage.at<double>(i, j) = getPixel(i, j) - getPixel(i, j - 1);
                else
                    dxImage.at<double>(i, j) = (getPixel(i, j + 1) - getPixel(i, j - 1)) / 2.0;
            }

            return CreateGroup(dxImage, dyImage);
        }

        Group<Mat, Mat> GetGaussDerivKernels(double sigma, double epsilon)
        {
            int halfSize = (int)ceil(sigma * sqrt(-2 * log(sqrt(2 * CV_PI) * sigma * epsilon)));
            int size = halfSize * 2 + 1;
            Mat dx(size, size, CV_64F), dy(size, size, CV_64F);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dx.at<double>(i, j) = Math::Gauss(i - halfSize, sigma) * 
                        Math::GaussDeriv(j - halfSize, sigma);
                    dy.at<double>(j, i) = dx.at<double>(i, j);
                }
            }

            normalize(dx, dx, 1.0, 0.0, NORM_L2);
            normalize(dy, dy, 1.0, 0.0, NORM_L2);

            return CreateGroup(dx, dy);
        }

        ArrayList<Mat> GetGaussDerivKernels(int orientNum, double sigma, double epsilon)
        {
            int halfSize = (int)ceil(sigma * sqrt(-2 * log(sqrt(2 * CV_PI) * sigma * epsilon)));
            int size = halfSize * 2 + 1;
            Mat dx(size, size, CV_64F), dy(size, size, CV_64F);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dx.at<double>(i, j) = Math::Gauss(i - halfSize, sigma) *
                        Math::GaussDeriv(j - halfSize, sigma);
                    dy.at<double>(j, i) = dx.at<double>(i, j);
                }
            }

            ArrayList<Mat> kernels(orientNum);
            for (int i = 0; i < orientNum; i++)
            {
                double orient = CV_PI / orientNum * i;
                kernels[i] = dx * cos(orient) + dy * sin(orient);
                normalize(kernels[i], kernels[i], 1.0, 0.0, NORM_L2);
            }

            return kernels;
        }

        Group<Mat, Mat> GetGradient(const Mat& image, double sigma)
        {
            Group<Mat, Mat> kernel = GetGaussDerivKernels(sigma, 1e-2);
            Mat dxImage, dyImage;
            filter2D(image, dxImage, CV_64F, kernel.Item1());
            filter2D(image, dyImage, CV_64F, kernel.Item2());

            Mat orientImage(image.rows, image.cols, CV_64F);
            for (int i = 0; i < image.rows; i++)
            {
                for (int j = 0; j < image.cols; j++)
                {
                    double orient = atan2(dyImage.at<double>(i, j), dxImage.at<double>(i, j));
                    while (orient >= CV_PI)
                        orient -= CV_PI;
                    while (orient < 0)
                        orient += CV_PI;

                    orientImage.at<double>(i, j) = orient;
                }
            }

            Mat powerImage(image.rows, image.cols, CV_64F);
            for (int i = 0; i < image.rows; i++)
                for (int j = 0; j < image.cols; j++)
                    powerImage.at<double>(i, j) = sqrt(
                    dyImage.at<double>(i, j) * dyImage.at<double>(i, j) +
                    dxImage.at<double>(i, j) * dxImage.at<double>(i, j));

            return CreateGroup(powerImage, orientImage);
        }

        ArrayList<Mat> GetGaussDerivChannels(const Mat& image, int orientNum, double sigma)
        {
            ArrayList<Mat> kernels = GetGaussDerivKernels(orientNum, sigma, 1e-2);
            ArrayList<Mat> channels(orientNum);

            for (int i = 0; i < orientNum; i++)
            {
                filter2D(image, channels[i], CV_64F, kernels[i]);
                channels[i] = abs(channels[i]);
            }

            return channels;
        }

        ArrayList<Mat> GetOrientChannels(const Mat& image, int orientNum, double sigma)
        {
            Group<Mat, Mat> gradient = GetGradient(image, sigma);
            Mat& powerImage = gradient.Item1();
            Mat& orientImage = gradient.Item2();
            int height = image.rows, width = image.cols;
            double orientBinSize = CV_PI / orientNum;

            ArrayList<Mat> orientChannels;
            for (int i = 0; i < orientNum; i++)
                orientChannels.Add(Mat::zeros(height, width, CV_64F));

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int o = (int)(orientImage.at<double>(i, j) / orientBinSize);
                    if (o < 0)
                        o = 0;
                    if (o >= orientNum)
                        o = orientNum - 1;

                    for (int k = -1; k <= 1; k++)
                    {
                        int newO = o + k;
                        double oRatio = 1 - abs((newO + 0.5) * orientBinSize - 
                            orientImage.at<double>(i, j)) / orientBinSize;
                        if (oRatio < 0)
                            oRatio = 0;

                        if (newO == -1)
                            newO = orientNum - 1;
                        if (newO == orientNum)
                            newO = 0;

                        orientChannels[newO].at<double>(i, j) += 
                            powerImage.at<double>(i, j) * oRatio;
                    }
                }
            }

            return orientChannels;
        }

        Mat GetStudentTKernel(Size ksize, double sigma, double theta, double lambd, 
            double gamma, double psi, int ktype)
        {
            double sigma_x = sigma;
            double sigma_y = sigma/gamma;
            int nstds = 3;
            int xmin, xmax, ymin, ymax;
            double c = cos(theta), s = sin(theta);

            if (ksize.width > 0)
                xmax = ksize.width/2;
            else
                xmax = cvRound(std::max(fabs(nstds * sigma_x * c), fabs(nstds * sigma_y * s)));

            if (ksize.height > 0)
                ymax = ksize.height/2;
            else
                ymax = cvRound(std::max(fabs(nstds * sigma_x * s), fabs(nstds * sigma_y * c)));

            xmin = -xmax;
            ymin = -ymax;

            CV_Assert(ktype == CV_32F || ktype == CV_64F);

            Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
            double scale = 1.0 / (CV_PI * sigma);
            double ex = 1.0 / (sigma_x * sigma_x);
            double ey = 1.0 / (sigma_y * sigma_y);
            double cscale = CV_PI * 2.0 / lambd;

            for (int y = ymin; y <= ymax; y++)
            {
                for (int x = xmin; x <= xmax; x++)
                {
                    double xr = x * c + y * s;
                    double yr = -x * s + y * c;
                    double v = scale / (1 + ex * xr * xr + ey * yr * yr) * cos(cscale * xr + psi);

                    if (ktype == CV_32F)
                        kernel.at<float>(ymax - y, xmax - x) = (float)v;
                    else
                        kernel.at<double>(ymax - y, xmax - x) = v;
                }
            }
        
            return kernel;
        }

        ArrayList<Mat> GetGaborChannels(const Mat& image, int orientNum, double sigma, double lambda)
        {
            int ksize = sigma * 6 + 1;
            ArrayList<Mat> gaborChannels(orientNum);

            for (int i = 0; i < orientNum; i++)
            {
                Mat kernel = getGaborKernel(Size(ksize, ksize), sigma, 
                    CV_PI / orientNum * i, lambda, 1, 0);

                filter2D(image, gaborChannels[i], CV_64F, kernel);
                gaborChannels[i] = abs(gaborChannels[i]);
            }

            return gaborChannels;
        }


        //////////////////////////////////////////////////////////////////////////
        // APIs for LoG and DoG
        //////////////////////////////////////////////////////////////////////////

        // http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
        Mat GetLoGKernel(int ksize, double sigma, int ktype)
        {
            CV_Assert(ksize > 0 && ksize % 2 != 0);
            CV_Assert(ktype == CV_64F || ktype == CV_32F);

            int halfSize = ksize / 2;
            Mat kernel(ksize, ksize, ktype);

            double scale = -1 / (CV_PI * pow(sigma, 4));
            for (int i = 0; i < ksize; i++)
            {
                for (int j = i; j < ksize; j++)
                {
                    double y = i - halfSize, x = j - halfSize;
                    double tmp = -(x * x + y * y) / (2 * sigma * sigma);
                    double value = scale * (1 + tmp) * exp(tmp);

                    if (ktype == CV_64F)
                    {
                        kernel.at<double>(i, j) = value;
                        kernel.at<double>(j, i) = kernel.at<double>(i, j);
                    }
                    else
                    {
                        kernel.at<float>(i, j) = (float)value;
                        kernel.at<float>(j, i) = kernel.at<float>(i, j);
                    }
                }
            }

            return kernel;
        }

        ArrayList<Mat> GetLoGPyramid(const Mat& image, const ArrayList<double>& sigmas)
        {
            size_t sigmaNum = sigmas.Count();
            ArrayList<Mat> LoGPyramid(sigmaNum);

            for (int i = 0; i < sigmaNum; i++)
            {
                CV_Assert(sigmas[i] >= 0);

                int ksize = (int)(sigmas[i] * 6 + 1);
                if (ksize % 2 == 0)
                    ksize++;

                Mat kernel = GetLoGKernel(ksize, sigmas[i], CV_64F);
                filter2D(image, LoGPyramid[i], CV_64F, kernel);
                LoGPyramid[i] = abs(LoGPyramid[i]) * pow(sigmas[i], 4); // pow(sigmas[i], 4) normalizes the integral
            }

            return LoGPyramid;
        }

        ArrayList<Mat> GetDoGPyramid(const Mat& image, const ArrayList<double>& sigmas)
        {
            size_t sigmaNum = sigmas.Count();
            ArrayList<Mat> GaussianPyramid(sigmaNum + 1);

            image.convertTo(GaussianPyramid[0], CV_64F);
            for (int i = 0; i < sigmaNum; i++)
            {
                CV_Assert(sigmas[i] >= 0);

                int ksize = (int)(sigmas[i] * 6 + 1);
                if (ksize % 2 == 0)
                    ksize++;

                Mat kernel = getGaussianKernel(ksize, sigmas[i], CV_64F);
                sepFilter2D(image, GaussianPyramid[i + 1], CV_64F, kernel, kernel);
            }

            ArrayList<Mat> DoGPyramid;
            for (int i = 1; i <= sigmaNum; i++)
                DoGPyramid[i - 1] = GaussianPyramid[i] - GaussianPyramid[i - 1];

            return DoGPyramid;
        }


        //////////////////////////////////////////////////////////////////////////
        // Others
        //////////////////////////////////////////////////////////////////////////

        ArrayList<Point> SampleOnGrid(size_t height, size_t width, size_t numPerDirection)
        {
            int heightStep = height / numPerDirection, 
                widthStep = width / numPerDirection;
            ArrayList<Point> points;

            for (int i = heightStep / 2; i < height; i += heightStep)
                for (int j = widthStep / 2; j < width; j += widthStep)
                    points.Add(Point(j, i));

            return points;
        }

        ArrayList<Point> SampleFromPoints(const ArrayList<Point>& points, size_t samplingNum)
        {
            size_t pointNum = points.Count();
            samplingNum = min(pointNum, samplingNum);

            ArrayList<Group<double, Group<size_t, size_t>>> distances(pointNum * (pointNum - 1) / 2);
            unordered_set<Point, PointHash> pivots;

            int counter = 0;
            for (size_t i = 0; i < pointNum; i++)
            {
                for (size_t j = i + 1; j < pointNum; j++)
                {
                    double distance = EulerDistance(points[i], points[j]);
                    distances[counter++] = CreateGroup(distance, CreateGroup(i, j));
                }
                pivots.insert(points[i]);
            }
            sort(distances.begin(), distances.end());

            int ptr = 0;
            while (pivots.size() > samplingNum)
            {
                Group<size_t, size_t> idxPair = distances[ptr++].Item2();
                Group<Point, Point> pointPair = CreateGroup(points[idxPair.Item1()], points[idxPair.Item2()]);

                if (pivots.find(pointPair.Item1()) != pivots.end() &&
                    pivots.find(pointPair.Item2()) != pivots.end())
                    pivots.erase(pointPair.Item2());
            }

            ArrayList<Point> results;
            for (auto pivot : pivots)
                results.Add(pivot);

            return results;
        }

        ArrayList<Mat> GetSurfaceCurvature(const Mat& Z)
        {
            Mat X(Z.size(), CV_64F), Y(Z.size(), CV_64F);

            for (int i = 0; i < Z.rows; i++)
            for (int j = 0; j < Z.cols; j++)
            {
                X.at<double>(i, j) = j;
                Y.at<double>(i, j) = i;
            }

            return GetSurfaceCurvature(X, Y, Z);
        }

        ArrayList<Mat> GetSurfaceCurvature(const Mat& X, const Mat& Y, const Mat& Z)
        {
            // first derivatives
            auto tmp = GetDiscreteGradient(X);
            Mat Xu = tmp.Item1(), Xv = tmp.Item2();

            tmp = GetDiscreteGradient(Y);
            Mat Yu = tmp.Item1(), Yv = tmp.Item2();

            tmp = GetDiscreteGradient(Z);
            Mat Zu = tmp.Item1(), Zv = tmp.Item2();

            // second derivatives
            tmp = GetDiscreteGradient(Xu);
            Mat Xuu = tmp.Item1(), Xuv = tmp.Item2();

            tmp = GetDiscreteGradient(Xv);
            Mat Xvv = tmp.Item2();

            tmp = GetDiscreteGradient(Yu);
            Mat Yuu = tmp.Item1(), Yuv = tmp.Item2();

            tmp = GetDiscreteGradient(Yv);
            Mat Yvv = tmp.Item2();

            tmp = GetDiscreteGradient(Zu);
            Mat Zuu = tmp.Item1(), Zuv = tmp.Item2();

            tmp = GetDiscreteGradient(Zv);
            Mat Zvv = tmp.Item2();

            // reshape 2D arrays into vectors
            Xu = Xu.reshape(1, Xu.rows * Xu.cols);
            Yu = Yu.reshape(1, Yu.rows * Yu.cols);
            Zu = Zu.reshape(1, Zu.rows * Zu.cols);
            Xv = Xv.reshape(1, Xv.rows * Xv.cols);
            Yv = Yv.reshape(1, Yv.rows * Yv.cols);
            Zv = Zv.reshape(1, Zv.rows * Zv.cols);
            Xuu = Xuu.reshape(1, Xuu.rows * Xuu.cols);
            Yuu = Yuu.reshape(1, Yuu.rows * Yuu.cols);
            Zuu = Zuu.reshape(1, Zuu.rows * Zuu.cols);
            Xuv = Xuv.reshape(1, Xuv.rows * Xuv.cols);
            Yuv = Yuv.reshape(1, Yuv.rows * Yuv.cols);
            Zuv = Zuv.reshape(1, Zuv.rows * Zuv.cols);
            Xvv = Xvv.reshape(1, Xvv.rows * Xvv.cols);
            Yvv = Yvv.reshape(1, Yvv.rows * Yvv.cols);
            Zvv = Zvv.reshape(1, Zvv.rows * Zvv.cols);

            Mat Du, Dv, Duu, Duv, Dvv;
            Du = Xu.clone(); hconcat(Du, Yu, Du); hconcat(Du, Zu, Du);
            Dv = Xv.clone(); hconcat(Dv, Yv, Dv); hconcat(Dv, Zv, Dv);
            Duu = Xuu.clone(); hconcat(Duu, Yuu, Duu); hconcat(Duu, Zuu, Duu);
            Duv = Xuv.clone(); hconcat(Duv, Yuv, Duv); hconcat(Duv, Zuv, Duv);
            Dvv = Xvv.clone(); hconcat(Dvv, Yvv, Dvv); hconcat(Dvv, Zvv, Dvv);

            // first fundamental coeffecients of the surface(E, F, G)
            Mat E;
            reduce(Du.mul(Du), E, 1, CV_REDUCE_SUM);

            Mat F;
            reduce(Du.mul(Dv), F, 1, CV_REDUCE_SUM);

            Mat G;
            reduce(Dv.mul(Dv), G, 1, CV_REDUCE_SUM);

            Mat m(Du.size(), CV_64F);
            for (int i = 0; i < Du.rows; i++)
                Du.row(i).cross(Dv.row(i)).copyTo(m.row(i));

            Mat p;
            reduce(m.mul(m), p, 1, CV_REDUCE_SUM);
            sqrt(p, p);
            Mat n = m / repeat(p, 1, m.cols);

            // second fundamental coeffecients of the surface(L, M, N)
            Mat L;
            reduce(Duu.mul(n), L, 1, CV_REDUCE_SUM);

            Mat M;
            reduce(Duv.mul(n), M, 1, CV_REDUCE_SUM);

            Mat N;
            reduce(Dvv.mul(n), N, 1, CV_REDUCE_SUM);

            // Gaussian curvature
            Mat K = (L.mul(N) - M.mul(M)) / (E.mul(G) - F.mul(F));
            K = K.reshape(1, Z.rows);

            // mean curvature
            Mat H = (E.mul(N) + G.mul(L) - 2 * F.mul(M)) / (2 * (E.mul(G) - F.mul(F)));
            H = H.reshape(1, Z.rows);

            // principal curvatures
            Mat dif;
            sqrt(max(H.mul(H) - K, 0), dif);
            Mat Pmax = H + dif;
            Mat Pmin = H - dif;

            ArrayList<Mat> result;
            result.Add(K);
            result.Add(H);
            result.Add(Pmax);
            result.Add(Pmin);

            return result;
        }
	}
}
}