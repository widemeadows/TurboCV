using Exocortex.DSP;
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

        public static Mat<T> CopyMakeBorder<T>(Mat<T> src, int paddingTop, int paddingBottom, 
            int paddingLeft, int paddingRight, T paddingValue) where T : IComparable<T>
        {
            Mat<T> dst = new Mat<T>(src.Rows + paddingTop + paddingBottom, 
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

        public static Mat<double> Convolve2D(Mat<byte> src, Mat<double> kernel, byte paddingValue = 0)
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

        public static Complex[] FFT2D(Mat<double> src)
        {
            Complex[] data = new Complex[src.Rows * src.Cols];

            for (int i = 0; i < src.Rows; i++)
                for (int j = 0; j < src.Cols; j++)
                    data[i * src.Cols + j] = new Complex(src[i, j], 0);

            Fourier.FFT2(data, src.Cols, src.Rows, FourierDirection.Forward);

            return data;
        }

        public static Complex[] FFT2D(Mat<byte> src)
        {
            Complex[] data = new Complex[src.Rows * src.Cols];

            for (int i = 0; i < src.Rows; i++)
                for (int j = 0; j < src.Cols; j++)
                    data[i * src.Cols + j] = new Complex(src[i, j], 0);

            Fourier.FFT2(data, src.Cols, src.Rows, FourierDirection.Forward);

            return data;
        }

        public static Mat<double> IFFT2D(Complex[] data, int rows, int cols)
        {
            Fourier.FFT2(data, cols, rows, FourierDirection.Backward);

            Mat<double> result = new Mat<double>(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = data[i * cols + j].Re / (rows * cols);

            return result;
        }

        public static int AlignToPow2(int num)
        {
            int len = 0;

            while ((1 << len) < num)
                len++;

            return 1 << len;
        }

        public static Mat<double> Filter2D(Mat<byte> src, Mat<double> kernel, byte paddingValue = 0)
        {
            if (src.Rows % 2 == 1 || src.Cols % 2 == 1 || 
                kernel.Rows % 2 != 1 || kernel.Cols % 2 != 1)
                return new Mat<double>(src.Size);

            Size extSize = new Size(src.Rows + kernel.Rows - 1, src.Cols + kernel.Cols - 1);
            extSize.Width = AlignToPow2(extSize.Width);
            extSize.Height = AlignToPow2(extSize.Height);

            Mat<byte> srcExt = CopyMakeBorder(src,
                0, extSize.Height - src.Rows,
                0, extSize.Width - src.Cols,
                paddingValue);

            Mat<double> kernelExt = CopyMakeBorder(kernel,
                0, extSize.Height - kernel.Rows,
                0, extSize.Width - kernel.Cols,
                0);

            Complex[] srcFFT = FFT2D(srcExt);
            Complex[] kernelFFT = FFT2D(kernelExt);
            for (int i = 0; i < srcFFT.Length; i++)
                srcFFT[i] *= kernelFFT[i];

            Mat<double> tmp = IFFT2D(srcFFT, srcExt.Rows, srcExt.Cols);
            Mat<double> result = new Mat<double>(src.Size);
            for (int i = 0; i < src.Rows; i++)
                for (int j = 0; j < src.Cols; j++)
                    result[i, j] = tmp[i + kernel.Rows / 2, j + kernel.Cols / 2];

            return result;
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

        public static Mat<byte> GetBoundingBox(Mat<byte> src)
        {
            int minX = src.Cols - 1, maxX = 0,
                minY = src.Rows - 1, maxY = 0;

            for (int i = 0; i < src.Rows; i++)
            {
                for (int j = 0; j < src.Cols; j++)
                {
                    if (src[i, j] != byte.MinValue)
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

        public static List<Point> GetEdgels(Mat<byte> src)
        {
            List<Point> edgels = new List<Point>();

            for (int i = 0; i < src.Rows; i++)
            {
                for (int j = 0; j < src.Cols; j++)
                {
                    if (src[i, j] != byte.MinValue)
                        edgels.Add(new Point(j, i));
                }
            }

            return edgels;
        }

        public static List<Mat<byte>> SplitViaOrientation(Mat<byte> src, int orientNum = 4)
        {
            int sigma = 9, lambda = 24, ksize = sigma * 6 + 1;
            List<Mat<double>> tmp = new List<Mat<double>>(orientNum);

            for (int i = 0; i < orientNum; i++)
            {
                Mat<double> kernel = Filter.GetGaborKernel(new Size(ksize, ksize), sigma, 
                    Math.PI / orientNum * i, lambda, 1, 0);

                tmp.Add(ImgProc.Filter2D(src, kernel));
            }

            List<Mat<byte>> channels = new List<Mat<byte>>(orientNum);
            for (int i = 0; i < orientNum; i++)
            {
                Mat<byte> channel = new Mat<byte>(src.Size);
                channel.Set(byte.MinValue);

                channels.Add(channel);
            }

            for (int i = 0; i < src.Rows; i++)
            {
                for (int j = 0; j < src.Cols; j++)
                {
                    if (src[i, j] != byte.MinValue)
                    {
                        double maxResponse = double.MinValue;
                        int o = -1;

                        for (int k = 0; k < orientNum; k++)
                        {
                            tmp[k][i, j] = Math.Abs(tmp[k][i, j]);

                            if (tmp[k][i, j] > maxResponse)
                            {
                                maxResponse = tmp[k][i, j];
                                o = k;
                            }
                        }

                        channels[o][i, j] = src[i, j];
                    }
                }
            }

            return channels;
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

        public static Mat<double> GetGaborKernel(Size ksize, double sigma, double theta, 
            double lambd, double gamma, double psi)
        {
            double sigma_x = sigma, sigma_y = sigma / gamma;
            int nstds = 3;
            int xmin, xmax, ymin, ymax;
            double c = Math.Cos(theta), s = Math.Sin(theta);
            
            if (ksize.Width > 0)
                xmax = ksize.Width / 2;
            else
                xmax = (int)Math.Max(Math.Abs(nstds * sigma_x * c), Math.Abs(nstds * sigma_y * s));
            
            if (ksize.Height > 0)
                ymax = ksize.Height / 2;
            else
                ymax = (int)Math.Max(Math.Abs(nstds * sigma_x * s), Math.Abs(nstds * sigma_y * c));
               
            xmin = -xmax;
            ymin = -ymax;
            
            Mat<double> kernel = new Mat<double>(ymax - ymin + 1, xmax - xmin + 1);
            double scale = 1 / (2 * Math.PI * sigma_x * sigma_y);
            double ex = -0.5 / (sigma_x * sigma_x);
            double ey = -0.5 / (sigma_y * sigma_y);
            double cscale = Math.PI * 2 / lambd;

            for (int y = ymin; y <= ymax; y++)
            {
                for (int x = xmin; x <= xmax; x++)
                {
                    double xr = x * c + y * s;
                    double yr = -x * s + y * c;

                    double v = scale * Math.Exp(ex * xr * xr + ey * yr * yr) *
                        Math.Cos(cscale * xr + psi);

                    kernel[ymax - y, xmax - x] = v;
                }
            }

            return kernel;
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

        public static Tuple<Mat<double>, Mat<double>> GetGradient(Mat<byte> image, double sigma = 1.0)
        {
            Tuple<Mat<double>, Mat<double>> kernel = GetGradientKernel(sigma);
            Mat<double> dxImage = ImgProc.Convolve2D(image, kernel.Item1);
            Mat<double> dyImage = ImgProc.Convolve2D(image, kernel.Item2);

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
            Mat<double> tmp = ImgProc.Convolve2D(image, GetGaussianKernel(sigma));

            Mat<byte> result = new Mat<byte>(tmp.Size);
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Cols; j++)
                    result[i, j] = (byte)tmp[i, j];

            return result;
        }
    }
}
