
#ifndef Converter_hpp
#define Converter_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

namespace irotavg
{
    namespace Converter
    {
        std::vector<cv::Mat> descriptorsMatToVector(const cv::Mat &descriptors);
    }
}


#endif /* Converter_hpp */
