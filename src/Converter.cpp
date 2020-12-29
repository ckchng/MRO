#include "Converter.hpp"

using namespace irotavg;

//// TODO: move to an static fucntion in some util class
std::vector<cv::Mat> Converter::descriptorsMatToVector(const cv::Mat &descriptors)
{
    std::vector<cv::Mat> out;
    const int n = descriptors.rows;
    out.reserve(n);
    for (int j=0; j<n; j++)
        out.push_back(descriptors.row(j));

    return out;
}

