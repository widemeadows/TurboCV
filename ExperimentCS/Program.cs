using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Turbo.System.CS;

namespace ExperimentCS
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Uri uri = new Uri("00001.png", UriKind.Relative);
            BitmapSource bmp = new PngBitmapDecoder(uri, BitmapCreateOptions.None,
                BitmapCacheOption.Default).Frames[0];

            Mat<byte> mat = BitmapSourceToMat(bmp);
            mat = GHOG.Preprocess(mat, new Size(256, 256));

            Mat<double> kernel = new Mat<double>(new Size(3, 3));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    kernel[i, j] = 1.0 / 9;
            Mat<double> tmp = ImgProc.Convolve(mat, kernel);
            for (int i = 0; i < tmp.Rows; i++)
                for (int j = 0; j < tmp.Cols; j++)
                    mat[i, j] = (byte)tmp[i, j];

            ImageBox box = new ImageBox(MatToBitmapSource(mat));
            box.ShowDialog();
        }

        public static Mat<byte> BitmapSourceToMat(BitmapSource source)
        {
            WriteableBitmap writeableBmp = new WriteableBitmap(source);
            byte[] rawData = new byte[writeableBmp.PixelWidth * writeableBmp.PixelHeight];
            writeableBmp.CopyPixels(rawData, writeableBmp.PixelWidth, 0);

            Mat<byte> image = new Mat<byte>(new Size(writeableBmp.PixelHeight, writeableBmp.PixelWidth));
            for (int i = 0; i < writeableBmp.PixelHeight; i++)
                for (int j = 0; j < writeableBmp.PixelWidth; j++)
                    image[i, j] = rawData[i * writeableBmp.PixelWidth + j];

            return image;
        }

        public static BitmapSource MatToBitmapSource(Mat<byte> mat)
        {
            byte[] buffer = new byte[mat.Rows * mat.Cols];
            for (int i = 0; i < mat.Rows; i++)
                for (int j = 0; j < mat.Cols; j++)
                    buffer[i * mat.Cols + j] = mat[i, j];

            return BitmapSource.Create(mat.Cols, mat.Rows, 96, 96, PixelFormats.Gray8,
                null, buffer, mat.Cols); 
        }
    }
}
