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

#include <vector>
#include <map>
using namespace std;

#define EPT_UCHAR   1
#define EPT_FLOAT   2

typedef unsigned char uchar;

struct EXPORT_API NativeMat
{
    NativeMat(int rows, int cols, int type);
    NativeMat(const NativeMat& other);

    ~NativeMat();

    NativeMat& operator=(const NativeMat& other);

    template<typename T>
    T& at(int row, int col)
    {
        return ((T**)m)[row][col];
    }

    template<typename T>
    const T& at(int row, int col) const
    {
        return ((T**)m)[row][col];
    }

    void clear();

    void* m;
    int rows, cols, type;
};

struct EXPORT_API NativePoint
{
    NativePoint(int x, int y)
    {
        this->x = x;
        this->y = y;
    }

    int x, y;
};

typedef vector<pair<vector<NativePoint>, NativeMat>> NativeInfo;

EXPORT_API NativeInfo PerformHitmap(const NativeMat& image, bool thinning);
EXPORT_API vector<NativeInfo> PerformHitmap(const vector<NativeMat>& images, bool thinning);