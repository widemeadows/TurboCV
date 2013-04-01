// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the EXPORT_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// EXPORT_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef EXPORT_EXPORTS
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __declspec(dllimport)
#endif

#include "../System/Type.h"
using namespace System;

template<typename T>
struct EXPORT_API Matrix
{
    Matrix(int rows, int cols)
    {
        m = new T*[rows];
        for (int i = 0; i < rows; i++)
            m[i] = new T[cols];
    
        this->rows = rows;
        this->cols = cols;
    }

    Matrix(const Matrix& other)
    {
        this->rows = other.rows;
        this->cols = other.cols;

        m = new T*[rows];
        for (int i = 0; i < rows; i++)
        {
            m[i] = new T[cols];
            for (int j = 0; j < cols; j++)
                m[i][j] = other[i][j];
        }
    }

    ~Matrix()
    {
        clear();
    }

    Matrix& operator=(const Matrix& other)
    {
        if (m == other.m)
            return this;

        if (rows != other.rows || cols != other.cols)
        {
            clear();

            rows = other.rows;
            cols = other.cols;

            m = new T*[rows];
            for (int i = 0; i < rows; i++)
                m[i] = new T[cols];
        }
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = other.m[i][j];

        return this;
    }

    T& at(int row, int col)
    {
        return m[row][col];
    }

    const T& at(int row, int col) const
    {
        return m[row][col];
    }

    void clear()
    {
        for (int i = 0; i < rows; i++)
            delete[] m[i];
        delete[] m;
    }

    T** m;
    int rows, cols;
};

struct EXPORT_API Position
{
    Position(int x, int y)
    {
        this->x = x;
        this->y = y;
    }

    int x, y;
};

typedef unsigned char uchar;

EXPORT_API vector<pair<vector<Position>, Matrix<float>>> PerformHitmap(
    const Matrix<uchar>& image, bool thinning);

