#ifndef RECONSTRUCTER
#define RECONSTRUCTER

#include "homographyManager.h"
#include "graphCutsMinimizer.h"

class Reconstructer
{
public:
    Reconstructer(){ 
    	homoManager = HomographyManager();
    	minimizer 	= GraphCutsMinimizer();
    }

    void reconstruct(cv::Mat& imTgt, cv::Mat& imSrc, cv::Size size, std::vector <DataForMinimizer>& datas, std::vector<cv::KeyPoint>& matched1, std::vector<cv::KeyPoint>& matched2);

    
protected:
    HomographyManager homoManager;
    GraphCutsMinimizer minimizer;

    int m_homoNum = 3;

};
#endif // RECONSTRUCTER