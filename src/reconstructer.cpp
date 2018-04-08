#include "reconstructer.h"

void Reconstructer::reconstruct(std::vector < std::vector<cv::Point2f> > blobs, std::vector<cv::KeyPoint>& matched1, 
								std::vector<cv::KeyPoint>& matched2)
{
	homoManager.findSeveralHomo(m_homoNum, matched1, matched2);

	homoManager.removeWrongHomo();

	for(auto blob: blobs)
	{

	}

}