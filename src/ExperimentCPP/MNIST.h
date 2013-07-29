#pragma once

#include "../System/System.h"
#include <cv.h>
#include <highgui.h>
#include <cstdio>
using namespace TurboCV::System;
using namespace cv;
using namespace std;

inline unsigned int ByteArrayToInt(unsigned char* byteArr, bool bigEndian = false)
{
    unsigned int result = 0;

    if (bigEndian)
    {
        result += (unsigned int)byteArr[3];
        result += (unsigned int)byteArr[2] << 8;
        result += (unsigned int)byteArr[1] << 16;
        result += (unsigned int)byteArr[0] << 24;
    }
    else
    {
        result += (unsigned int)byteArr[3] << 24;
        result += (unsigned int)byteArr[2] << 16;
        result += (unsigned int)byteArr[1] << 8;
        result += (unsigned int)byteArr[0];
    }

    return result;
}

inline ArrayList<Mat> ReadMnistImages(const TString& imageFilePath)
{
    FILE* file = fopen(imageFilePath, "rb");
    unsigned char magic[4], nImage[4], nRow[4], nCol[4];

    fread(magic, 1, 4, file);
    fread(nImage, 1, 4, file);
    fread(nRow, 1, 4, file);
    fread(nCol, 1, 4, file);

    int rows = ByteArrayToInt(nRow, true),
        cols = ByteArrayToInt(nCol, true);
    unsigned char* img = new unsigned char[rows * cols];
    ArrayList<Mat> images;

    for (int i = ByteArrayToInt(nImage, true) - 1; i >= 0; i--)
    {
        fread(img, 1, rows * cols, file);

        Mat mat(rows, cols, CV_8U);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                mat.at<uchar>(i, j) = img[i * cols + j];

        images.Add(mat);
    }

    delete[] img;
    fclose(file);
    return images;
}

inline ArrayList<int> ReadMnistLabels(const TString& labelFilePath)
{
    FILE* file = fopen(labelFilePath, "rb");
    unsigned char magic[4], nItem[4];

    fread(magic, 1, 4, file);
    fread(nItem, 1, 4, file);

    ArrayList<int> labels;
    unsigned char byte;

    for (int i = ByteArrayToInt(nItem, true) - 1; i >= 0; i--)
    {
        fread(&byte, 1, 1, file);
        labels.Add(byte);
    }

    fclose(file);
    return labels;
}