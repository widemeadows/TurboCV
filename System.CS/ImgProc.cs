using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace System.CS
{
    public class ImgProc
    {
        public static Mat<T> CopyMakeBorder<T>(Mat<T> src, int paddingTop, int paddingBottom, 
            int paddingLeft, int paddingRight, T paddingValue)
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

        public static Mat<double> Convolve(Mat<byte> src, Mat<double> kernel, byte paddingValue = 0)
        {
            if (kernel.Rows % 2 != 1 || kernel.Cols % 2 != 1)
                return new Mat<double>(src.Rows, src.Cols);

            int halfKernelHeight = kernel.Rows / 2, halfKernelWidth = kernel.Cols / 2;
            int top = halfKernelHeight, bottom = top + src.Rows,
                left = halfKernelWidth, right = left + src.Cols;

            Mat<byte> tmp = CopyMakeBorder(src, halfKernelHeight, halfKernelHeight,
                halfKernelWidth, halfKernelWidth, paddingValue);
            Mat<double> dst = new Mat<double>(src.Rows, src.Cols);

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
                                    dst[i, j] = 0;
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
                                    dst[i, j] = 0;
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

        public static Mat<byte> Reverse(Mat<byte> src)
        {
            Mat<byte> dst = new Mat<byte>(src.Rows, src.Cols);

            for (int i = 0; i < src.Rows; i++)
                for (int j = 0; j < src.Cols; j++)
                    dst[i, j] = (byte)(byte.MaxValue - src[i, j]);

            return dst;
        }
    }
}
