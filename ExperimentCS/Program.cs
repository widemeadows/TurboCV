using System;
using System.Collections.Generic;
using System.IO;
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
            DirectoryInfo rootDir = new DirectoryInfo("oracles");
            List<int> train_labels = new List<int>(),
                      test_labels = new List<int>();
            List<Mat<byte>> train_images = new List<Mat<byte>>(),
                            test_images = new List<Mat<byte>>();

            int index = 1;
            foreach (var dir in rootDir.GetDirectories())
            {
                int counter = 0;
                foreach (var file in dir.GetFiles())
                {
                    BitmapSource bmp = new PngBitmapDecoder(new Uri(file.FullName),
                        BitmapCreateOptions.None, BitmapCacheOption.Default).Frames[0];

                    FormatConvertedBitmap formatedBmp = new FormatConvertedBitmap();
                    formatedBmp.BeginInit();
                    formatedBmp.Source = bmp;
                    formatedBmp.DestinationFormat = PixelFormats.Gray8;
                    formatedBmp.EndInit();

                    if (counter % 3 < 2)
                    {
                        train_images.Add(BitmapSourceToMat(formatedBmp));
                        train_labels.Add(index);
                    }
                    else
                    {
                        test_images.Add(BitmapSourceToMat(formatedBmp));
                        test_labels.Add(index);
                    }

                    counter++;
                }

                index++;
            }

            List<GlobalFeatureVec> train_features = new List<GlobalFeatureVec>(),
                                   test_features = new List<GlobalFeatureVec>();

            for (int i = 0; i < train_images.Count; i++)
                train_features.Add(new GlobalFeatureVec());
            for (int i = 0; i < test_images.Count; i++)
                test_features.Add(new GlobalFeatureVec());

            Parallel.For(0, train_images.Count, (i) =>
            {
                train_features[i] = GHOG.GetFeatureWithoutPreprocess(train_images[i]);
            });
            Parallel.For(0, test_images.Count, (i) =>
            {
                test_features[i] = GHOG.GetFeatureWithoutPreprocess(test_images[i]);
            });

            Console.WriteLine(KNN(train_features, train_labels, test_features, test_labels));

            //Uri uri = new Uri("00001.png", UriKind.Relative);
            //BitmapSource bmp = new PngBitmapDecoder(uri, BitmapCreateOptions.None,
            //    BitmapCacheOption.Default).Frames[0];

            //FormatConvertedBitmap formatedBmp = new FormatConvertedBitmap();
            //formatedBmp.BeginInit();
            //formatedBmp.Source = bmp;
            //formatedBmp.DestinationFormat = PixelFormats.Gray8;
            //formatedBmp.EndInit();

            //Mat<byte> mat = GHOG.Preprocess(BitmapSourceToMat(formatedBmp), new Size(256, 256));

            //ImageBox box = new ImageBox(MatToBitmapSource(mat));
            //box.ShowDialog();
        }

        public static double KNN(
            List<GlobalFeatureVec> train_features, 
            List<int> train_labels,
            List<GlobalFeatureVec> test_features,
            List<int> test_labels,
            int K = 4)
        {
            List<int> predict_labels = new List<int>();
            for (int i = 0; i < test_labels.Count; i++)
                predict_labels.Add(0);

            Parallel.For(0, test_labels.Count, (i) =>
            {
                List<Tuple<double, int>> distAndIndexes = new List<Tuple<double, int>>();
                for (int j = 0; j < train_labels.Count; j++)
                {
                    distAndIndexes.Add(Tuple.Create(
                        train_features[j].GetDistance(test_features[i]), j));
                }

                distAndIndexes.Sort();

                Dictionary<int, int> dict = new Dictionary<int, int>();
                for (int j = 0; j < K; j++)
                {
                    int label = train_labels[distAndIndexes[j].Item2];

                    if (dict.ContainsKey(label))
                        dict[label]++;
                    else
                        dict[label] = 1;
                }

                int max = -1;
                foreach (var item in dict)
                {
                    if (item.Value > max)
                    {
                        predict_labels[i] = item.Key;
                        max = item.Value;
                    }
                }
            });

            double precision = 0;
            for (int i = 0; i < predict_labels.Count; i++)
                if (predict_labels[i] == test_labels[i])
                    precision++;

            return precision / predict_labels.Count;
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
