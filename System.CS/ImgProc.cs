using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Turbo.System.CS
{
    public class ImgProc
    {
        public static Mat<byte> Reverse(Mat<byte> src)
        {
            Mat<byte> dst = new Mat<byte>(src.Size);

            for (int i = 0; i < src.Rows; i++)
                for (int j = 0; j < src.Cols; j++)
                    dst[i, j] = (byte)(byte.MaxValue - src[i, j]);

            return dst;
        }

        public static Mat<byte> CopyMakeBorder(Mat<byte> src, int paddingTop, int paddingBottom, 
            int paddingLeft, int paddingRight, byte paddingValue)
        {
            Mat<byte> dst = new Mat<byte>(src.Rows + paddingTop + paddingBottom, 
                src.Cols + paddingLeft + paddingRight);

            dst.Set(paddingValue);
            
            int top = paddingTop, bottom = paddingTop + src.Rows,
                left = paddingLeft, right = paddingLeft + src.Cols;
            for (int i = top; i < bottom; i++)
                for (int j = left; j < right; j++)
                    dst[i, j] = src[i - top, j - left];

            return dst;
        }

        public static Mat<byte> Resize(Mat<byte> src, Size size)
        {
            double xRatio = (double)size.Width / (double)src.Cols;
            double yRatio = (double)size.Height / (double)src.Rows;

            Mat<int> count = new Mat<int>(size);
            Mat<double> tmp = new Mat<double>(size);

            for (int i = 0; i < src.Rows; i++)
            {
                for (int j = 0; j < src.Cols; j++)
                {
                    int r = (int)(i * yRatio), c = (int)(j * xRatio);
                    if (r >= tmp.Rows)
                        r = tmp.Rows - 1;
                    if (c >= tmp.Cols)
                        c = tmp.Cols - 1;

                    count[r, c]++;
                    tmp[r, c] += src[i, j];
                }
            }

            for (int i = 0; i < tmp.Rows; i++)
                for (int j = 0; j < tmp.Cols; j++)
                    if (count[i, j] > 0)
                        tmp[i, j] /= count[i, j];

            Mat<byte> dst = new Mat<byte>(tmp.Size);

            for (int i = 0; i < tmp.Rows; i++)
            {
                for (int j = 0; j < tmp.Cols; j++)
                {
                    if (count[i, j] > 0)
                    {
                        dst[i, j] = (byte)tmp[i, j];
                    }
                    else
                    {
                        int r = 0, c = 0;

                        if (yRatio < 0)
                        {
                            r = i;
                        }
                        else
                        {
                            int upBound = (int)((int)(i / yRatio) * yRatio);
                            int downBound = (int)(((int)(i / yRatio) + 1) * yRatio);

                            if (i - upBound < downBound - i)
                                r = upBound;
                            else
                                r = downBound;
                        }

                        if (xRatio < 0)
                        {
                            c = j;
                        }
                        else
                        {
                            int leftBound = (int)((int)(j / xRatio) * xRatio);
                            int rightBound = (int)(((int)(j / xRatio) + 1) * xRatio);

                            if (j - leftBound < rightBound - j)
                                c = leftBound;
                            else
                                c = rightBound;
                        }

                        if (r >= tmp.Rows)
                            r = tmp.Rows - 1;
                        if (c >= tmp.Cols)
                            c = tmp.Cols - 1;

                        dst[i, j] = (byte)tmp[r, c];
                    }
                }
            }

            return dst;
        }

        public static Mat<double> Convolve(Mat<byte> src, Mat<double> kernel, byte paddingValue = 0)
        {
            if (kernel.Rows % 2 != 1 || kernel.Cols % 2 != 1)
                return new Mat<double>(src.Size);

            int halfKernelHeight = kernel.Rows / 2, halfKernelWidth = kernel.Cols / 2;
            int top = halfKernelHeight, bottom = top + src.Rows,
                left = halfKernelWidth, right = left + src.Cols;

            Mat<byte> tmp = CopyMakeBorder(src, halfKernelHeight, halfKernelHeight,
                halfKernelWidth, halfKernelWidth, paddingValue);
            Mat<double> dst = new Mat<double>(src.Size);

            for (int i = top; i < bottom; i++)
            {
                for (int j = left; j < right; j++)
                {
                    double sum = 0;
                    for (int m = i - halfKernelHeight; m <= i + halfKernelHeight; m++)
                        for (int n = j - halfKernelWidth; n <= j + halfKernelWidth; n++)
                            sum += tmp[m, n] * kernel[m - i + halfKernelHeight, n - j + halfKernelWidth];

                    dst[i - top, j - left] = sum;
                }
            }

            return dst;
        }
    }

    public class BinaryImgProc
    {
        public static Mat<byte> Thin(Mat<byte> src, int iterations = 100) 
        {
            Mat<byte> dst = src.Clone();

            for (int n = 0; n < iterations; n++) 
            {
                Mat<byte> prev = dst.Clone();
                for (int i = 0; i < src.Rows; i++) 
                {
                    for (int j = 0; j < src.Cols; j++) 
                    {
                        if (prev[i, j] == byte.MaxValue) 
                        {
                            int ap = 0;
                            int p2 = (i == 0) ? 0 : prev[i - 1, j];
                            int p3 = (i == 0 || j == src.Cols - 1) ? 0 : prev[i - 1, j + 1];
                            if (p2 == byte.MinValue && p3 == byte.MaxValue) 
                                ap++;

                            int p4 = (j == src.Cols - 1) ? 0 : prev[i, j + 1];
                            if (p3 == byte.MinValue && p4 == byte.MaxValue) 
                                ap++;

                            int p5 = (i == src.Rows - 1 || j == src.Cols - 1) ? 0 : prev[i + 1, j + 1];
                            if (p4 == byte.MinValue && p5 == byte.MaxValue) 
                                ap++;

                            int p6 = (i == src.Rows - 1) ? 0 : prev[i + 1, j];
                            if (p5 == byte.MinValue && p6 == byte.MaxValue) 
                                ap++;

                            int p7 = (i == src.Rows - 1 || j == 0) ? 0 : prev[i + 1, j - 1];
                            if (p6 == byte.MinValue && p7 == byte.MaxValue) 
                                ap++;

                            int p8 = (j == 0) ? 0 : prev[i, j - 1];
                            if (p7 == byte.MinValue && p8 == byte.MaxValue) 
                                ap++;

                            int p9 = (i == 0 || j == 0) ? 0 : prev[i - 1, j - 1];
                            if (p8 == byte.MinValue && p9 == byte.MaxValue) 
                                ap++;
                            if (p9 == byte.MinValue && p2 == byte.MaxValue) 
                                ap++;

                            if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / byte.MaxValue > 1 && 
                                (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / byte.MaxValue < 7)
                                if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
                                    dst[i, j] = byte.MinValue;
                        }
                    }
                }

                prev = dst.Clone();
                for (int i = 0; i < src.Rows; i++) 
                {
                    for (int j = 0; j < src.Cols; j++)
                    {
                        if (prev[i, j] == byte.MaxValue)
                        {
                            int ap = 0;
                            int p2 = (i == 0) ? 0 : prev[i - 1, j];
                            int p3 = (i == 0 || j == src.Cols - 1) ? 0 : prev[i - 1, j + 1];
                            if (p2 == byte.MinValue && p3 == byte.MaxValue) 
                                ap++;

                            int p4 = (j == src.Cols - 1) ? 0 : prev[i, j + 1];
                            if (p3 == byte.MinValue && p4 == byte.MaxValue) 
                                ap++;

                            int p5 = (i == src.Rows - 1 || j == src.Cols - 1) ? 0 : prev[i + 1, j + 1];
                            if (p4 == byte.MinValue && p5 == byte.MaxValue) 
                                ap++;

                            int p6 = (i == src.Rows - 1) ? 0 : prev[i + 1, j];
                            if (p5 == byte.MinValue && p6 == byte.MaxValue) 
                                ap++;

                            int p7 = (i == src.Rows - 1 || j == 0) ? 0 : prev[i + 1, j - 1];
                            if (p6 == byte.MinValue && p7 == byte.MaxValue) 
                                ap++;

                            int p8 = (j == 0) ? 0 : prev[i, j - 1];
                            if (p7 == byte.MinValue && p8 == byte.MaxValue) 
                                ap++;

                            int p9 = (i == 0 || j == 0) ? 0 : prev[i - 1, j - 1];
                            if (p8 == byte.MinValue && p9 == byte.MaxValue) 
                                ap++;
                            if (p9 == byte.MinValue && p2 == byte.MaxValue) 
                                ap++;

                            if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / byte.MaxValue > 1 && 
                                (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / byte.MaxValue < 7) 
                                if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) 
                                    dst[i, j] = byte.MinValue;
                        }
                    }
                }
            }

            return dst;
        }

        public static Mat<byte> Clean(Mat<byte> src, int points = 1)
        {
            Mat<byte> dst = src.Clone();
            Mat<byte> cur = src.Clone();

            int[] dy = { -1, -1, -1, 0, 0, 1, 1, 1 };
            int[] dx = { -1, 0, 1, -1, 1, -1, 0, 1 };

            for (int i = 0; i < cur.Rows; i++)
            {
                for (int j = 0; j < cur.Cols; j++)
                {
                    if (cur[i, j] == byte.MaxValue)
                    {
                        Queue<Point> q = new Queue<Point>();
                        List<Point> component = new List<Point>();

                        q.Enqueue(new Point(j, i));
                        cur[i, j] = 0;

                        while (q.Count != 0)
                        {
                            Point front = q.Dequeue();
                            component.Add(front);

                            for (int k = 0; k < 8; k++)
                            {
                                int newY = front.Y + dy[k], newX = front.X + dx[k];

                                if (newY >= 0 && newY < cur.Rows && 
                                    newX >= 0 && newX < cur.Cols &&
                                    cur[newY, newX] == byte.MaxValue)
                                {
                                    q.Enqueue(new Point(newX, newY));
                                    cur[newY, newX] = 0;
                                }
                            }
                        }

                        if (component.Count <= points)
                        {
                            foreach (Point point in component)
                            {
                                dst[point.Y, point.X] = 0;
                            }
                        }
                    }
                }
            }

            return dst;
        }

        public static Mat<byte> GetBoundingBox(Mat<byte> src, byte backgroundGrayLevel = byte.MinValue)
        {
            int minX = src.Cols - 1, maxX = 0,
                minY = src.Rows - 1, maxY = 0;

            for (int i = 0; i < src.Rows; i++)
            {
                for (int j = 0; j < src.Cols; j++)
                {
                    if (src[i, j] != backgroundGrayLevel)
                    {
                        minX = Math.Min(minX, j);
                        maxX = Math.Max(maxX, j);
                        minY = Math.Min(minY, i);
                        maxY = Math.Max(maxY, i);
                    }
                }
            }

            Mat<byte> dst = new Mat<byte>(maxY - minY + 1, maxX - minX + 1);
            for (int i = minY; i <= maxY; i++)
                for (int j = minX; j <= maxX; j++)
                    dst[i - minY, j - minX] = src[i, j];

            return dst;
        }
    }

    public class Filter
    {
        public static double Gauss(double x, double sigma)
        {
            return Math.Exp(-Math.Pow(x, 2.0) / (2 * Math.Pow(sigma, 2.0))) / (sigma * Math.Sqrt(2 * Math.PI));
        }

        public static double GaussDeriv(double x, double sigma)
        {
            return -x * Gauss(x, sigma) / Math.Pow(sigma, 2);
        }

        public static Tuple<Mat<double>, Mat<double>> GetGradientKernel(double sigma, double epsilon = 1e-2)
        {
            int halfSize = (int)Math.Ceiling(sigma * Math.Sqrt(-2 * Math.Log(Math.Sqrt(2 * Math.PI) * sigma * epsilon)));
            int size = halfSize * 2 + 1;
            double sum = 0, root;
            Mat<double> dx = new Mat<double>(size, size), dy = new Mat<double>(size, size);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dy[j, i] = dx[i, j] = Gauss(i - halfSize, sigma) * GaussDeriv(j - halfSize, sigma);
                    sum += dx[i, j] * dx[i, j];
                }
            }

            root = Math.Sqrt(sum);
            if (root > 0)
            {
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        dx[i, j] /= root;
                        dy[i, j] /= root;
                    }
                }
            }
            
            return Tuple.Create(dx, dy);
        }

        public static Mat<double> GetGaussianKernel(double sigma, double epsilon = 1e-2)
        {
            int halfSize = (int)Math.Ceiling(sigma * Math.Sqrt(-2 * Math.Log(Math.Sqrt(2 * Math.PI) * sigma * epsilon)));
            int size = halfSize * 2 + 1;
            double sum = 0, root;
            Mat<double> kernel = new Mat<double>(size, size);

            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    kernel[i, j] = Gauss(i - halfSize, sigma) * Gauss(j - halfSize, sigma);

            root = Math.Sqrt(sum);
            if (root > 0)
                for (int i = 0; i < size; i++)
                    for (int j = 0; j < size; j++)
                        kernel[i, j] /= root;

            return kernel;
        }

        public static Tuple<Mat<double>, Mat<double>> GetGradient(Mat<byte> image, double sigma = 1.0)
        {
            Tuple<Mat<double>, Mat<double>> kernel = GetGradientKernel(sigma);
            Mat<double> dxImage = ImgProc.Convolve(image, kernel.Item1);
            Mat<double> dyImage = ImgProc.Convolve(image, kernel.Item2);

            Mat<double> orientImage = new Mat<double>(image.Size);
            for (int i = 0; i < image.Rows; i++)
            {
                for (int j = 0; j < image.Cols; j++)
                {
                    double orient = Math.Atan2(dyImage[i, j], dxImage[i, j]);
                    while (orient >= Math.PI)
                        orient -= Math.PI;
                    while (orient < 0)
                        orient += Math.PI;

                    orientImage[i, j] = orient;
                }
            }

            Mat<double> powerImage = new Mat<double>(image.Size);
            for (int i = 0; i < image.Rows; i++)
                for (int j = 0; j < image.Cols; j++)
                    powerImage[i, j] = Math.Sqrt(
                        dyImage[i, j] * dyImage[i, j] + dxImage[i, j] * dxImage[i, j]);

            return Tuple.Create(powerImage, orientImage);
        }

        public static List<Mat<double>> GetOrientChannels(Mat<byte> sketchImage, int orientNum)
        {
            Tuple<Mat<double>, Mat<double>> gradient = GetGradient(sketchImage);
            Mat<double> powerImage = gradient.Item1;
            Mat<double> orientImage = gradient.Item2;
            int height = sketchImage.Rows, width = sketchImage.Cols;
            double orientBinSize = Math.PI / orientNum;

            List<Mat<double>> orientChannels = new List<Mat<double>>();
            for (int i = 0; i < orientNum; i++)
                orientChannels.Add(new Mat<double>(sketchImage.Size));

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int o = (int)(orientImage[i, j] / orientBinSize);
                    if (o < 0)
                        o = 0;
                    if (o >= orientNum)
                        o = orientNum - 1;

                    for (int k = -1; k <= 1; k++)
                    {
                        int newO = o + k;
                        double oRatio = 1 - Math.Abs((newO + 0.5) * orientBinSize - 
                            orientImage[i, j]) / orientBinSize;
                        if (oRatio < 0)
                            oRatio = 0;
            
                        if (newO == -1)
                            newO = orientNum - 1;
                        if (newO == orientNum)
                            newO = 0;

                        orientChannels[newO][i, j] += powerImage[i, j] * oRatio;
                    }
                }
            }

            return orientChannels;
        }

        public static Mat<byte> Blur(Mat<byte> image, double sigma = 1.0)
        {
            Mat<double> tmp = ImgProc.Convolve(image, GetGaussianKernel(sigma));

            Mat<byte> result = new Mat<byte>(tmp.Size);
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Cols; j++)
                    result[i, j] = (byte)tmp[i, j];

            return result;
        }
    }
}
