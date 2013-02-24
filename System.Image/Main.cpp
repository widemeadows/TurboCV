#include "Feature.h"
#include <cv.h>
#include <highgui.h>
using namespace System::Image;
using namespace cv;

int main()
{
    Feature* feature = new HOG();
    
    feature->GetFeature(imread("00001.png", CV_LOAD_IMAGE_GRAYSCALE), true);

    delete feature;
}