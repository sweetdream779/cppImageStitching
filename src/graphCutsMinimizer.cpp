#include "graphCutsMinimizer.h"


float euclidNorm(cv::Vec3b& color1, cv::Vec3b& color2)
{
	float dr = 1.0*color1[0] - 1.0*color2[0];
	float dg = 1.0*color1[1] - 1.0*color2[1];
	float db = 1.0*color1[2] - 1.0*color2[2];

	return std::sqrt(dr*dr + dg*dg + db*db);
}

void getVtVs(cv::Point2f& pt, cv::Mat& homo, cv::Mat& imTgt, cv::Mat& imScr, cv::Vec3b& Vt, cv::Vec3b& Vs)
{
	std::vector<cv::Point2f> pts_in(1);
    pts_in[0] = pt;
    std::vector<cv::Point2f> pts_out(1);
    cv::perspectiveTransform(pts_in, pts_out, homo);

    Vt = imTgt.at<cv::Vec3b>(cv::Point(pts_out[0].x, pts_out[0].y));
    Vs = imScr.at<cv::Vec3b>(cv::Point(pts_out[0].x, pts_out[0].y));
}

int smoothFn(int p1, int p2, int l1, int l2)
{
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}

double smoothFn2(int p1, int p2, int l1, int l2)
{
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}


void GridGraph_DArraySArray(std::vector<cv::Mat>& imsTgt, cv::Mat& imSrc,
							DataForMinimizer& d, std::vector<cv::Mat>& homoSet, int width, int height, 
							int num_pixels, int num_labels, std::vector<int>& result)
{

	result.resize(num_pixels);   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
		{
			if(!d.needTransforms[i])
			{
				data[i*num_labels+l] = 0;
				continue;
			}
			cv::Vec3b Vt, Vs;
			getVtVs(d.points[i], homoSet[l], imsTgt[l], imSrc, Vt, Vs);
			data[i*num_labels+l] = euclidNorm(Vt, Vs);
		}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = 1;


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);
		gc->setDataCost(data);
		//gc->setSmoothCost(smooth);
		gc->setSmoothCost(&smoothFn);
		printf("\nBefore optimization energy is %lld",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %lld",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] smooth;
	delete [] data;

}

void GraphCutsMinimizer::optimize(std::vector<cv::Mat> imsTgt, cv::Mat& imSrc, DataForMinimizer& data, std::vector<cv::Mat> homoSet, std::vector<cv::Mat> invHomoSet)
{
	int width = data.rect.width;
	int height = data.rect.height;
	int num_pixels = width*height;

	int num_labels = invHomoSet.size();
	
	// smoothness and data costs are set up using arrays
	GridGraph_DArraySArray(imsTgt, imSrc, data, homoSet, width, height, num_pixels, num_labels, data.homoIdxs);
}
