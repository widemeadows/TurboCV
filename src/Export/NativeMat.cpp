#include "stdafx.h"
#include "Export.h"

NativeMat::NativeMat(int rows, int cols, BasicType type)
{
    this->rows = rows;
    this->cols = cols;
    this->type = type;

    if (type == EPT_UCHAR)
    {
        m = new uchar*[rows];
        for (int i = 0; i < rows; i++)
            ((uchar**)m)[i] = new uchar[cols];
    }
    else
    {
        m = new float*[rows];
        for (int i = 0; i < rows; i++)
            ((float**)m)[i] = new float[cols];
    }
}

NativeMat::NativeMat(const NativeMat& other)
{
    this->rows = other.rows;
    this->cols = other.cols;
    this->type = other.type;

    if (this->type == EPT_UCHAR)
    {
        m = new uchar*[rows];
        for (int i = 0; i < rows; i++)
        {
            ((uchar**)m)[i] = new uchar[cols];
            for (int j = 0; j < cols; j++)
                ((uchar**)m)[i][j] = ((uchar**)other.m)[i][j];
        }
    }
    else
    {
        m = new float*[rows];
        for (int i = 0; i < rows; i++)
        {
            ((float**)m)[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                ((float**)m)[i][j] = ((float**)other.m)[i][j];
        }
    }
}

NativeMat::~NativeMat()
{
    clear();
}

NativeMat& NativeMat::operator=(const NativeMat& other)
{
    if (m == other.m)
        return *this;

    if (rows != other.rows || cols != other.cols || type != other.type)
    {
        clear();

        rows = other.rows;
        cols = other.cols;
        type = other.type;

        if (this->type == EPT_UCHAR)
        {
            m = new uchar*[rows];
            for (int i = 0; i < rows; i++)
                ((uchar**)m)[i] = new uchar[cols];
        }
        else
        {
            m = new float*[rows];
            for (int i = 0; i < rows; i++)
                ((float**)m)[i] = new float[cols];
        }
    }

    if (this->type == EPT_UCHAR)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ((uchar**)m)[i][j] = ((uchar**)other.m)[i][j];
    }
    else
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ((float**)m)[i][j] = ((float**)other.m)[i][j];
    }

    return *this;
}

void NativeMat::clear()
{
    if (type == EPT_UCHAR)
    {
        for (int i = 0; i < rows; i++)
            delete[] ((uchar**)m)[i];
        delete[] (uchar**)m;
    }
    else
    {
        for (int i = 0; i < rows; i++)
            delete[] ((float**)m)[i];
        delete[] (float**)m;
    }
}