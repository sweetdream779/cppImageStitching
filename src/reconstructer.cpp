#include "reconstructer.h"

cv::Vec3b getColor(const cv::Mat& imSrc, const cv::Point2f& pt, cv::Mat homo)
{
	/*std::vector<cv::Point2f> pts_in(1);
    pts_in[0] = pt;
    std::vector<cv::Point2f> pts_out(1);
    cv::perspectiveTransform(pts_in, pts_out, homo);*/

	return imSrc.at<cv::Vec3b>(cv::Point(pt.x, pt.y));
}

void Reconstructer::reconstruct(cv::Mat imTgt, cv::Mat imSrc, cv::Size sizeSrc, cv::Size sizeTgt, std::vector <DataForMinimizer>& datas, 
								std::vector<cv::KeyPoint>& matched1, std::vector<cv::KeyPoint>& matched2)
{
	//src to tgt
	homoManager.findSeveralHomo(m_homoNum, matched1, matched2);

	homoManager.removeWrongHomo();

	//homos map target image to src, нужно найти координаты точек в которые бы отображалась наш блоб из таргет, а потом взять эти точки
	//homoManager.setTransformedTgtImgs(imTgt, sizeTgt);
	homoManager.setTransformedSrcImgs(imSrc, cv::Size(imTgt.cols + imSrc.cols, imTgt.rows));

	std::vector<cv::Mat> homoSet = homoManager.getHomoSet();
	std::vector<cv::Mat> invHomoSet = homoManager.getInvHomoSet();
	//std::vector<cv::Mat> transformedTgtImgs = homoManager.getTransformedTgtImgs();	
	std::vector<cv::Mat> transformedSrcImgs = homoManager.getTransformedSrcImgs();		


	for(auto& data: datas)
	{
		minimizer.optimize(transformedSrcImgs, imTgt, data, homoSet);
		for(int i = 0; i < data.points.size(); ++i)
		{
			if(data.needTransforms[i])
			{
				int homoInd = data.homoIdxs[i];
				cv::Vec3b newValue = getColor(transformedSrcImgs[homoInd], data.points[i], invHomoSet[homoInd]);
				imTgt.at<cv::Vec3b>(cv::Point(data.points[i].x, data.points[i].y)) = newValue;
			}
		}
		

		for(int i = 0; i < m_homoNum; ++i)
		{
			cv::Mat roi1(transformedSrcImgs[i], Rect(0, 0,  imTgt.cols, imTgt.rows));
    		imTgt.copyTo(roi1);
    		cv::imshow("ddddd", transformedSrcImgs[i]);
    		cv::waitKey(0);
		}
	}
}