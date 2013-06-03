﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Turbo.System.CS
{
    public class GlobalFeatureVec
    {
        public void Append(double value)
        {
            _vec.Add(value);
        }

        public double GetDistance(GlobalFeatureVec other)
        {
            if (_vec.Count != other._vec.Count)
                return double.NaN;

            double sum = 0;
            for (int i = 0; i < _vec.Count; i++)
                sum += Math.Abs(_vec[i] - other._vec[i]);

            return sum;
        }

        public double this[int i]
        {
            get
            {
                return _vec[i];
            }
            set
            {
                _vec[i] = value;
            }
        }

        public int Count
        {
            get
            {
                return _vec.Count;
            }
        }

        private List<double> _vec = new List<double>();
    }

    public class GHOG
    {
        public static Mat<byte> Preprocess(Mat<byte> src, Size size, bool thinning = true)
        {
            Mat<byte> revImage = ImgProc.Reverse(src);

            Mat<byte> cleanedImage = BinaryImgProc.Clean(revImage, 3);

            Mat<byte> boundingBox = BinaryImgProc.GetBoundingBox(revImage);

            int widthPadding = 0, heightPadding = 0;
            if (boundingBox.Rows < boundingBox.Cols)
                heightPadding = (boundingBox.Cols - boundingBox.Rows) / 2;
            else
                widthPadding = (boundingBox.Rows - boundingBox.Cols) / 2;
            Mat<byte> squareImage = ImgProc.CopyMakeBorder(boundingBox, 
                heightPadding, heightPadding, widthPadding, widthPadding, 0);

            Size scaledSize = new Size((int)(size.Height - 2 * size.Height / 16.0),
                (int)(size.Width - 2 * size.Width / 16.0));
            Mat<byte> scaledImage = ImgProc.Resize(squareImage, scaledSize);

            heightPadding = (size.Height - scaledSize.Height) / 2;
            widthPadding = (size.Width - scaledSize.Width) / 2; 
            Mat<byte> paddedImage = ImgProc.CopyMakeBorder(scaledImage, 
                heightPadding, heightPadding, widthPadding, widthPadding, 0);

            Mat<byte> finalImage = paddedImage.Clone();
            for (int i = 0; i < finalImage.Rows; i++)
            {
                for (int j = 0; j < finalImage.Cols; j++)
                {
                    if (finalImage[i, j] > 54)
                        finalImage[i, j] = byte.MaxValue;
                    else
                        finalImage[i, j] = 0;
                }
            }

            if (thinning)
                finalImage = BinaryImgProc.Thin(finalImage);

            return finalImage;
        }

        public static GlobalFeatureVec GetFeatureWithPreprocess(Mat<byte> src,
            bool normalize = true, int orientNum = 8, double blockRatio = 48.0 / 256.0)
        {
            src = Preprocess(src, new Size(256, 256));

            return GetFeatureWithoutPreprocess(src, normalize, orientNum, blockRatio);
        }

        public static GlobalFeatureVec GetFeatureWithoutPreprocess(Mat<byte> src, 
            bool normalize = true, int orientNum = 8, double blockRatio = 48.0 / 256.0)
        {    
            int blockSize = (int)(blockRatio * 256);
            List<Mat<double>> orientChannels = Filter.GetOrientChannels(src, orientNum);
            
            GlobalFeatureVec feature = new GlobalFeatureVec();
            for (int i = blockSize / 2 - 1; i < src.Rows; i += blockSize / 2)
            {
                for (int j = blockSize / 2 - 1; j < src.Cols; j += blockSize / 2)
                {
                    for (int k = 0; k < orientNum; k++)
                    {
                        double value = 0;

                        for (int m = i - blockSize; m <= i + blockSize; m++)
                        {
                            for (int n = j - blockSize; n <= j + blockSize; n++)
                            {
                                if (m < 0 || m >= src.Rows || n < 0 || n >= src.Cols)
                                    continue;

                                double weight = 1 - Math.Sqrt((m - i) * (m - i) + (n - j) * (n - j)) / blockSize;
                                if (weight < 0)
                                    weight = 0;

                                value += orientChannels[k][m, n] * weight;
                            }
                        }

                        feature.Append(value);
                    }
                }
            }

            if (normalize)
            {
                double sum = 0;
                for (int i = 0; i < feature.Count; i++)
                {
                    sum += feature[i] * feature[i];
                }

                double root = Math.Sqrt(sum);
                if (root > 0)
                {
                    for (int i = 0; i < feature.Count; i++)
                    {
                        feature[i] /= root;
                    }
                }
            }

            return feature;
        }
    }

    public class CONN
    {
        public static Mat<byte> Preprocess(Mat<byte> src, Size size, bool thinning = true)
        {
            Mat<byte> revImage = ImgProc.Reverse(src);

            Mat<byte> cleanedImage = BinaryImgProc.Clean(revImage, 3);

            Mat<byte> boundingBox = BinaryImgProc.GetBoundingBox(revImage);

            int widthPadding = 0, heightPadding = 0;
            if (boundingBox.Rows < boundingBox.Cols)
                heightPadding = (boundingBox.Cols - boundingBox.Rows) / 2;
            else
                widthPadding = (boundingBox.Rows - boundingBox.Cols) / 2;
            Mat<byte> squareImage = ImgProc.CopyMakeBorder(boundingBox,
                heightPadding, heightPadding, widthPadding, widthPadding, 0);

            Size scaledSize = new Size((int)(size.Height - 2 * size.Height / 16.0),
                (int)(size.Width - 2 * size.Width / 16.0));
            Mat<byte> scaledImage = ImgProc.Resize(squareImage, scaledSize);

            heightPadding = (size.Height - scaledSize.Height) / 2;
            widthPadding = (size.Width - scaledSize.Width) / 2;
            Mat<byte> paddedImage = ImgProc.CopyMakeBorder(scaledImage,
                heightPadding, heightPadding, widthPadding, widthPadding, 0);

            Mat<byte> finalImage = paddedImage.Clone();
            for (int i = 0; i < finalImage.Rows; i++)
            {
                for (int j = 0; j < finalImage.Cols; j++)
                {
                    if (finalImage[i, j] > 54)
                        finalImage[i, j] = byte.MaxValue;
                    else
                        finalImage[i, j] = 0;
                }
            }

            if (thinning)
                finalImage = BinaryImgProc.Thin(finalImage);

            return finalImage;
        }

        public static GlobalFeatureVec GetFeatureWithPreprocess(Mat<byte> src,
            bool normalize = true, int orientNum = 8, double blockRatio = 48.0 / 256.0)
        {
            src = Preprocess(src, new Size(256, 256));

            return GetFeatureWithoutPreprocess(src, normalize, orientNum, blockRatio);
        }

        public static GlobalFeatureVec GetFeatureWithoutPreprocess(Mat<byte> src,
            bool normalize = true, int orientNum = 8, double blockRatio = 48.0 / 256.0)
        {
            int blockSize = (int)(blockRatio * 256);
            int blockHalfSize = blockSize / 2;
            Mat<byte> hasVisited = new Mat<byte>(blockSize, blockSize);

            GlobalFeatureVec feature = new GlobalFeatureVec();
            for (int i = blockHalfSize - 1; i < src.Rows; i += blockSize)
            {
                for (int j = blockHalfSize - 1; j < src.Cols; j += blockSize)
                {
                    int top = i - (blockHalfSize - 1),
                        bottom = i + blockHalfSize,
                        left = j - (blockHalfSize - 1),
                        right = j + blockHalfSize;


                }
            }

            return feature;
        }
    }
}