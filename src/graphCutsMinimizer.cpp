#include "graphCutsMinimizer.h"

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
void GridGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);
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

	delete [] result;
	delete [] smooth;
	delete [] data;

}

void GraphCutsMinimizer::optimize(DataForMinimizer data, std::vector<cv::Mat> homoSet)
{
	int width = data.width;
	int height = data.height;
	int num_pixels = width*height;

	int num_labels = homoSet.size();

	// smoothness and data costs are set up using arrays
	GridGraph_DArraySArray(width, height, num_pixels, num_labels);
}
