#ifndef HOMOGRAPHYMANAGER
#define HOMOGRAPHYMANAGER

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/videoio.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <opencv2/flann.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct DataForMinimizer
{
	std::vector<cv::Point2f> points;
	std::vector<cv::Vec3b> colorsSrc;
	std::vector<std::vector<cv::Vec3b> > colorsTgt;
	std::vector<bool> needTransforms;
	std::vector<int> homoIdxs;
	cv::Rect rect;
};

class HomographyManager
{	
public:
    HomographyManager(){}
    HomographyManager (const HomographyManager& h_) = default;

    void setMatchedPoints(std::vector<cv::KeyPoint>& matched1,  std::vector<cv::KeyPoint>& matched2) 
    	{m_matched1 = matched1; m_matched2 = matched2;}
    void setTransformedImgs(cv::Mat& imSrc, cv::Size size);

    cv::Mat findOneHomo(std::vector<cv::KeyPoint>& inliers1,  std::vector<cv::KeyPoint>& inliers2, 
						std::vector<cv::KeyPoint>& outliers1, std::vector<cv::KeyPoint>& outliers2);
    cv::Mat findMainHomo(int k = 3);
    void    findSeveralHomo(const int num, std::vector<cv::KeyPoint> matched1, std::vector<cv::KeyPoint> matched2);
    void	removeWrongHomo();

    std::vector<cv::Point2f> transformPoints(std::vector<cv::Point2f>& pts_in, cv::Mat& homo);

    cv::Mat getInvertedH(){return m_H_inv;}
    std::vector<cv::Mat> getHomoSet(){return m_homographySet;}
    std::vector<cv::Mat> getInvHomoSet(){return m_invHomographySet;}
    std::vector<cv::Mat> getTransformedImgs(){return m_transformedImgs;}
private:
	double ransac_thresh = 2.5f; // RANSAC inlier threshold
	std::vector<cv::KeyPoint> m_matched1;
	std::vector<cv::KeyPoint> m_matched2;
	
	cv::Mat 				  m_H_inv;

	std::vector<cv::Mat>	  m_homographySet;
	std::vector<cv::Mat>	  m_invHomographySet;
	std::vector<cv::Mat>	  m_transformedImgs;
};

#endif // HOMOGRAPHYMANAGER