using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using TurboCV.System.CS;

namespace ExperimentCS
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            //DirectoryInfo rootDir = new DirectoryInfo("oracles");
            //List<int> train_labels = new List<int>(),
            //          test_labels = new List<int>();
            //List<Mat<byte>> train_images = new List<Mat<byte>>(),
            //                test_images = new List<Mat<byte>>();

            //int index = 1;
            //foreach (var dir in rootDir.GetDirectories())
            //{
            //    int counter = 0;
            //    foreach (var file in dir.GetFiles())
            //    {
            //        BitmapSource bmp = new PngBitmapDecoder(new Uri(file.FullName),
            //            BitmapCreateOptions.None, BitmapCacheOption.Default).Frames[0];

            //        FormatConvertedBitmap formatedBmp = new FormatConvertedBitmap();
            //        formatedBmp.BeginInit();
            //        formatedBmp.Source = bmp;
            //        formatedBmp.DestinationFormat = PixelFormats.Gray8;
            //        formatedBmp.EndInit();

            //        if (counter % 3 < 2)
            //        {
            //            train_images.Add(BitmapSourceToMat(formatedBmp));
            //            train_labels.Add(index);
            //        }
            //        else
            //        {
            //            test_images.Add(BitmapSourceToMat(formatedBmp));
            //            test_labels.Add(index);
            //        }

            //        counter++;
            //    }

            //    index++;
            //}

            //List<GlobalFeatureVec> train_features = new List<GlobalFeatureVec>(),
            //                       test_features = new List<GlobalFeatureVec>();

            //for (int i = 0; i < train_images.Count; i++)
            //    train_features.Add(new GlobalFeatureVec());
            //for (int i = 0; i < test_images.Count; i++)
            //    test_features.Add(new GlobalFeatureVec());

            //Parallel.For(0, train_images.Count, (i) =>
            //{
            //    train_features[i] = GHOG.GetFeatureWithoutPreprocess(train_images[i]);
            //});
            //Parallel.For(0, test_images.Count, (i) =>
            //{
            //    test_features[i] = GHOG.GetFeatureWithoutPreprocess(test_images[i]);
            //});

            //Console.WriteLine(KNN(train_features, train_labels, test_features, test_labels));

            Uri uri = new Uri("00006.png", UriKind.Relative);
            BitmapSource bmp = new PngBitmapDecoder(uri, BitmapCreateOptions.None,
                BitmapCacheOption.Default).Frames[0];

            FormatConvertedBitmap formatedBmp = new FormatConvertedBitmap();
            formatedBmp.BeginInit();
            formatedBmp.Source = bmp;
            formatedBmp.DestinationFormat = PixelFormats.Gray8;
            formatedBmp.EndInit();

            Mat<byte> preprocessed = Feature.Preprocess(BitmapSourceToMat(formatedBmp), new Size(256, 256));
            var channels = BinaryImgProc.SplitViaOrientation(preprocessed, 8);
            foreach (var channel in channels)
            {
                ImageBox box = new ImageBox(MatToBitmapSource(channel));
                box.ShowDialog();
            }

            //int sigma = 9, lambda = 24, ksize = sigma * 6 + 1;
            //Mat<double> filter = Filter.GetGaborKernel(new Size(ksize, ksize), sigma,
            //        0, lambda, 1, 0);
            //Mat<double> filter = new Mat<double>(5, 5);
            //for (int i = 0; i < filter.Rows; i++)
            //    for (int i = 0; i < filter.Cols; i++)
            //        filter[i, i] = 1.0 / (filter.Rows * filter.Cols);

            //Mat<double> tmp = ImgProc.Filter2D(preprocessed, filter);
            //Mat<byte> result = new Mat<byte>(tmp.Size);
            //double max = double.MinValue, min = double.MaxValue;
            //for (int i = 0; i < tmp.Rows; i++)
            //    for (int k = 0; k < tmp.Cols; k++)
            //    {
            //        if (tmp[i, k] > max)
            //            max = tmp[i, k];
            //        if (tmp[i, k] < min)
            //            min = tmp[i, k];
            //    }

            //for (int i = 0; i < tmp.Rows; i++)
            //    for (int k = 0; k < tmp.Cols; k++)
            //        result[i, k] = (byte)((tmp[i, k] - min) / (max - min) * 256);

            //ImageBox box = new ImageBox(MatToBitmapSource(result));
            //box.ShowDialog();
        }

        public static Mat<byte> BitmapSourceToMat(BitmapSource source)
        {
            WriteableBitmap writeableBmp = new WriteableBitmap(source);
            byte[] rawData = new byte[writeableBmp.PixelWidth * writeableBmp.PixelHeight];
            writeableBmp.CopyPixels(rawData, writeableBmp.PixelWidth, 0);

            Mat<byte> image = new Mat<byte>(writeableBmp.PixelHeight, writeableBmp.PixelWidth);
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
