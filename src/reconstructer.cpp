#include "reconstructer.h"

void Reconstructer::reconstruct(cv::Mat& imTgt, cv::Mat& imSrc, cv::Size sizeSrc, std::vector <DataForMinimizer>& datas, 
								std::vector<cv::KeyPoint>& matched1, std::vector<cv::KeyPoint>& matched2)
{
	homoManager.findSeveralHomo(m_homoNum, matched1, matched2);

	homoManager.removeWrongHomo();

	//homos map target image to src, нужно найти координаты точек в которые бы отображалась наш блоб из таргет, а потом взять эти точки
	homoManager.setTransformedImgs(imTgt, sizeSrc);

	for(auto& data: datas)
	{
		minimizer.optimize(homoManager.getTransformedImgs(), imSrc, data, homoManager.getHomoSet(), homoManager.getInvHomoSet());
	}

}