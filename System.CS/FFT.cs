using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Turbo.System.CS
{
    public struct Complex
    {
        public double real, imag;

        public Complex(double re, double im)
        {
            real = re;
            imag = im;
        }

        public double Magnitude()
        {
            return Math.Sqrt(real * real + imag * imag);
        }

        public double Phase()
        {
            return Math.Atan(imag / real);
        }

        public static Complex operator*(Complex c1, Complex c2)
        {
            double re = c1.real * c2.real - c1.imag * c2.imag;
            double im = c1.real * c2.imag + c1.imag * c2.real;
            return new Complex(re, im);
        }
    }

    public enum FourierDirection
    {
        Forward, Backward
    }

    public class Fourier
    {
        // Finding the power of 2 for the given number. 
        // e.g. For num = 1024 return 10.
        private static int findPower(long num)
        {
            int power = 0;

            while ((1 << power) < num)
                power++;

            return power;
        }

        private static void swap<T>(ref T u, ref T v)
        {
            T tmp = v;
            v = u;
            u = tmp;
        }

        /*-------------------------------------------------------------------------
            Perform a 2D FFT inplace given a complex 2D array.
            The dimensions of c must be powers of 2
        */
        public static Complex[,] FFT2D(Complex[,] c, FourierDirection dir)
        {
            int rows = c.GetLength(0), cols = c.GetLength(1);
            Complex[,] result = new Complex[rows, cols];

            double[] real = new double[cols];
            double[] imag = new double[cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    real[j] = c[i, j].real;
                    imag[j] = c[i, j].imag;
                }

                // Calling 1D FFT Function for Rows
                FFT1D(ref real, ref imag, dir);

                for (int j = 0; j < cols; j++)
                {
                    result[i, j].real = real[j];
                    result[i, j].imag = imag[j];
                }
            }

            real = new double[rows];
            imag = new double[rows];
            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    real[j] = result[j, i].real;
                    imag[j] = result[j, i].imag;
                }

                // Calling 1D FFT Function for Columns
                FFT1D(ref real, ref imag, dir);

                for (int j = 0; j < rows; j++)
                {
                    result[j, i].real = real[j];
                    result[j, i].imag = imag[j];
                }
            }

            return result;
        }

        /*-------------------------------------------------------------------------
            This computes an in-place complex-to-complex FFT
            re and im are the real and imaginary arrays of 2^m points.
            dir = 1 gives forward transform
            dir = -1 gives reverse transform
            
            Formula: forward
                      N-1
                      ---
                    1 \         - j k 2 pi n / N
            X(K) = --- > re(n) e                  = Forward transform
                    N /                            n=0..N-1
                      ---
                      n=0
            
            Formula: reverse
                     N-1
                     ---
                     \         j k 2 pi n / N
            X(n) =    > re(k) e                  = Inverse transform
                     /                            k=0..N-1
                     ---
                     k=0
            */
        private static void FFT1D(ref double[] x, ref double[] y, FourierDirection dir)
        {
            long n = x.Length, m = findPower(n);
            long i, i1, i2, j, k, l, l1, l2;
            double c1, c2, u1, u2, z;

            /* Do the power reversal */
            i2 = n >> 1;
            j = 0;
            for (i = 0; i < n - 1; i++)
            {
                if (i < j)
                {
                    swap(ref x[i], ref x[j]);
                    swap(ref y[i], ref y[j]);
                }
                k = i2;
                while (k <= j)
                {
                    j -= k;
                    k >>= 1;
                }
                j += k;
            }

            /* Compute the FFT */
            c1 = -1.0;
            c2 = 0.0;
            l2 = 1;
            for (l = 0; l < m; l++)
            {
                l1 = l2;
                l2 <<= 1;
                u1 = 1.0;
                u2 = 0.0;
                for (j = 0; j < l1; j++)
                {
                    for (i = j; i < n; i += l2)
                    {
                        i1 = i + l1;
                        double t1 = u1 * x[i1] - u2 * y[i1];
                        double t2 = u1 * y[i1] + u2 * x[i1];
                        x[i1] = x[i] - t1;
                        y[i1] = y[i] - t2;
                        x[i] += t1;
                        y[i] += t2;
                    }
                    z = u1 * c1 - u2 * c2;
                    u2 = u1 * c2 + u2 * c1;
                    u1 = z;
                }
                c2 = Math.Sqrt((1.0 - c1) / 2.0);
                if (dir == FourierDirection.Forward)
                    c2 = -c2;
                c1 = Math.Sqrt((1.0 + c1) / 2.0);
            }

            /* Scaling for reverse transform */
            if (dir == FourierDirection.Backward)
            {
                for (i = 0; i < n; i++)
                {
                    x[i] /= (double)n;
                    y[i] /= (double)n;
                }
            }
        }
    }
}
