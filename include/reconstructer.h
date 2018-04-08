#ifndef RECONSTRUCTER
#define RECONSTRUCTER

#include "homographyManager.h"
#include "graphCutsMinimizer.h"

class Reconstructer
{
public:
    Reconstructer(){ homoManager = HomographyManager();}

    void reconstruct(std::vector < std::vector<cv::Point2f> > blobs, std::vector<cv::KeyPoint>& matched1, std::vector<cv::KeyPoint>& matched2);

    
protected:
    HomographyManager homoManager;
    GraphCutsMinimizer minimizer;

    int m_homoNum = 3;

};
#endif // RECONSTRUCTER