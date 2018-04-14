#include "reconstructer.h"

cv::Vec3b getColor(const cv::Mat& imSrc, const cv::Point2f& pt, cv::Mat homo)
{
	/*std::vector<cv::Point2f> pts_in(1);
    pts_in[0] = pt;
    std::vector<cv::Point2f> pts_out(1);
    cv::perspectiveTransform(pts_in, pts_out, homo);*/

	return imSrc.at<cv::Vec3b>(cv::Point(pt.x, pt.y));
}

void gradientDomainFusion(std::vector<cv::Point2f>& maskPoints, cv::Mat& m, cv::Mat& s, cv::Mat& t, cv::Mat& res)
{
	int imH = m.rows;
	int imW = m.cols;
	int var = maskPoints.size();// number of variables to be solved
	std::cout<<"var: "<<var<<std::endl;
	cv::Mat im2var(imH, imW, CV_32FC1, Scalar(0)); //matrix that maps each pixel to a variable number
	int i = 0;
	for (int j = 0; j < var; ++j)
	{
	    im2var.at<float>(maskPoints[j].y, maskPoints[j].x) = i;
	    i++;
	}

	cv::Mat A(var, var, CV_32FC1, Scalar::all(0));
	cv::Mat b(var, 3, CV_32FC1, Scalar::all(0));

	int e = 0;
	int y, x, ind;
	unsigned char * p1;
	unsigned char * p2;
	unsigned char * p3;

	for (int j = 0; j < maskPoints.size(); ++j)
	{
	    y = maskPoints[j].y;
	    x = maskPoints[j].x;
	    //set up coefficients for A; 4(center)-1(left)-1(right)-1(up)-1(down)
	    ind = im2var.at<float>(y, x); 
	    A.at<float>(e, ind) = 4;

	    if (m.at<int>(y-1, x) == 1)
	    {
	        //if pixel is within the mask, take the gradient from the source image
	        
		    ind = im2var.at<float>(y-1, x); 
		    A.at<float>(e,ind) = -1;
		    p1 = s.ptr(y-1,x);
	        p2 = s.ptr(y,x);
		    b.at<float>(e,0) = b.at<float>(e,0) - ((float)p1[0]/255 - (float)p2[0]/255);
		    b.at<float>(e,1) = b.at<float>(e,1) - ((float)p1[1]/255 - (float)p2[1]/255);
		    b.at<float>(e,2) = b.at<float>(e,2) - ((float)p1[2]/255 - (float)p2[2]/255);	     
	          
	    }
	    else
	    {
	        //otherwise, directly take the pixel value from the target image	
	        p3 = t.ptr(y-1,x);  
		    b.at<float>(e,0) = b.at<float>(e,0) + (float)p3[0]/255;
		    b.at<float>(e,1) = b.at<float>(e,1) + (float)p3[1]/255;
		    b.at<float>(e,2) = b.at<float>(e,2) + (float)p3[2]/255;
		    
	    }
	    if (m.at<int>(y+1, x) == 1)
	    {
	    	
		    ind = im2var.at<float>(y+1, x); 
		    A.at<float>(e,ind) = -1;
		    p1 = s.ptr(y+1,x);
	        p2 = s.ptr(y,x);

		    b.at<float>(e,0) = b.at<float>(e,0) - ((float)p1[0]/255 - (float)p2[0]/255);
		    b.at<float>(e,1) = b.at<float>(e,1) - ((float)p1[1]/255 - (float)p2[1]/255);
		    b.at<float>(e,2) = b.at<float>(e,2) - ((float)p1[2]/255 - (float)p2[2]/255);
		    
	    }
	    else
	    {
	    	p3 = t.ptr(y+1,x);  
	    	b.at<float>(e,0) = b.at<float>(e,0) + (float)p3[0]/255;
		    b.at<float>(e,1) = b.at<float>(e,1) + (float)p3[1]/255;
		    b.at<float>(e,2) = b.at<float>(e,2) + (float)p3[2]/255;

	    }
	    if (m.at<int>(y, x-1) == 1)
	    {
	        
		    ind = im2var.at<float>(y, x-1);
		    A.at<float>(e, ind) = -1;
		    p1 = s.ptr(y,x-1);
	        p2 = s.ptr(y,x);

		    b.at<float>(e,0) = b.at<float>(e,0) - ((float)p1[0]/255 - (float)p2[0]/255);
		    b.at<float>(e,1) = b.at<float>(e,1) - ((float)p1[1]/255 - (float)p2[1]/255);
		    b.at<float>(e,2) = b.at<float>(e,2) - ((float)p1[2]/255 - (float)p2[2]/255);
		    
	    }
	    else
	    {
	    		p3 = t.ptr(y,x-1);
		        b.at<float>(e,0) = b.at<float>(e,0) + (float)p3[0]/255;
		        b.at<float>(e,1) = b.at<float>(e,1) + (float)p3[1]/255;
		        b.at<float>(e,2) = b.at<float>(e,2) + (float)p3[2]/255;
		    
	    }
	    if (m.at<int>(y, x+1) == 1)
	    {
	    	ind = im2var.at<float>(y,x+1); 
		    A.at<float>(e, ind) = -1;
		    p1 = s.ptr(y,x+1);
	        p2 = s.ptr(y,x);

		    b.at<float>(e,0) = b.at<float>(e,0) - ((float)p1[0]/255 - (float)p2[0]/255);
		    b.at<float>(e,1) = b.at<float>(e,1) - ((float)p1[1]/255 - (float)p2[1]/255);
		    b.at<float>(e,2) = b.at<float>(e,2) - ((float)p1[2]/255 - (float)p2[2]/255);
		    
	    }
	    else
	    {
	    	p3 = t.ptr(y,x+1);
		    b.at<float>(e,0) = b.at<float>(e,0) + (float)p3[0]/255;
		    b.at<float>(e,1) = b.at<float>(e,1) + (float)p3[1]/255;
		    b.at<float>(e,2) = b.at<float>(e,2) + (float)p3[2]/255;
	    }    
	    e++;
	}
	std::cout<<"System created."<<std::endl;
	cv::Mat v;
	//solve v for each rgb channel
	cv::solve(A, b, v);
	std::cout<<"System solved."<<std::endl;
	//cv::Mat v = A.inv() * b;

	e = 0;
	//copy the values over to the target image to the area to be blended
	for (int i = 0; i < var; ++i)
	{
	    y = maskPoints[i].y;
	    x = maskPoints[i].x;
	    
	    res.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(v.at<float>(e, 0)*255, v.at<float>(e, 1)*255, v.at<float>(e, 2)*255);
	    e++;
	}

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
		cv::Mat mask(imTgt.rows, imTgt.cols, CV_8UC1, cv::Scalar(0));
		cv::Mat mask2(imTgt.rows, imTgt.cols, CV_8UC3, cv::Scalar(0));
		cv::Mat im_s = Mat::zeros( cv::Size(imSrc.cols, imSrc.rows), CV_8UC3 );

		cv::Mat result = imTgt.clone();

		minimizer.optimize(transformedSrcImgs, imTgt, data, homoSet);
		for(int i = 0; i < data.points.size(); ++i)
		{
			int homoInd = data.homoIdxs[i];
			cv::Vec3b newValue = getColor(transformedSrcImgs[homoInd], data.points[i], invHomoSet[homoInd]);
			im_s.at<cv::Vec3b>(cv::Point(data.points[i].x, data.points[i].y)) = newValue;
			
			if(data.needTransforms[i])
			{
				result.at<cv::Vec3b>(cv::Point(data.points[i].x, data.points[i].y)) = newValue;
			}
		}
		for(cv::Point2f& pt: data.maskPoints){
			mask.at<int>(cv::Point(pt.x, pt.y)) = 1;
		}

		//gradient domain fusion
		cv::imshow("before", result);
		gradientDomainFusion(data.maskPoints, mask, im_s, result, result);

		cv::imshow("after", result);
	}
}