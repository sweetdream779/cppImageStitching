#include "reconstructer.h"

void Reconstructer::reconstruct(std::vector <DataForMinimizer> datas, std::vector<cv::KeyPoint>& matched1, 
								std::vector<cv::KeyPoint>& matched2)
{
	homoManager.findSeveralHomo(m_homoNum, matched1, matched2);

	homoManager.removeWrongHomo();

	for(auto& data: datas)
	{
		minimizer.optimize(data, homoManager.getHomoSet());
	}

}