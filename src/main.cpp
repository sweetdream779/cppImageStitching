#include "myStitcher.h"
#include "reconstructer.h"

cv::Mat getSegmentation(const cv::Mat src){
    cv::Mat seg = cv::imread("/home/irina/Desktop/im3_seg.png", 0);
    return seg;
}


void FindBlobs(const cv::Mat &img, std::vector < std::vector<cv::Point2f> >& blobs, int borderX)
{
    blobs.clear();
    cv::Mat binary;
    cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY);

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
            std::cout<<rect.x<<" "<<rect.x+rect.width<<std::endl;
            if(rect.x <= borderX && rect.x+rect.width >= borderX)
            {
                std::vector <cv::Point2f> blob;

                for(int i=rect.y; i < (rect.y+rect.height); i++) {
                    int *row2 = (int*)label_image.ptr(i);
                    for(int j=rect.x; j < (rect.x+rect.width); j++) {
                        if(row2[j] != label_count) {
                            continue;
                        }

                        blob.push_back(cv::Point2f(j,i));
                    }
                }

                if(blob.size() > 10)
                    blobs.push_back(blob);
            }
            else
                std::cout<<"Missed"<<std::endl;

            label_count++;
        }
    }
}

int main(int argc, char **argv){
    Mat image1,image2,res,vis;

	image1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    image2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); 
    
    //Ptr<KAZE> akaze = KAZE::create();
    //akaze->setThreshold(akaze_thresh);

    //Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
    //detector->setThreshold(100);
    
    //Ptr<GFTTDetector> detector=GFTTDetector::create();

    Ptr<SURF> detector = SURF::create( 400 );
    Ptr<BriefDescriptorExtractor> extractor = BriefDescriptorExtractor::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    cv::Mat seg1 = cv::imread("/home/irina/Desktop/im1_seg.png", 0);
    cv::Mat seg2 = cv::imread("/home/irina/Desktop/im3_seg.png", 0);

    seg1 = resizeImg(seg1);
    seg2 = resizeImg(seg2);

    int borderX1 = image1.cols - 1;

    int borderX2;
    std::vector<cv::KeyPoint> matched1, matched2;
    MyStitcher stitcher(detector, matcher, extractor);
    res=stitcher.stitch(image1, image2, matched1, matched2, borderX2);

    //find connected components on ipm
    std::vector < std::vector<cv::Point2f > > blobs1;
    FindBlobs(seg1, blobs1, borderX1);
    std::cout<<"Blobs on joint from image1: "<<blobs1.size()<<std::endl;


    std::vector < std::vector<cv::Point2f> > blobs2;
    FindBlobs(seg2, blobs2, borderX2);
    std::cout<<"Blobs on joint from image2: "<<blobs2.size()<<std::endl;

    Reconstructer reconstructer; 
    //pass first homo too

    if(blobs1.size() > 0 && blobs2.size() == 0)
    {
    	//если на первом, то удалить блобы
        reconstructer.reconstruct(blobs1, matched1, matched2);

    }
    if(blobs2.size() > 0 && blobs1.size() == 0)
    {
		//если на втором и не границе, то добавить блоб
        reconstructer.reconstruct(blobs2, matched1, matched2);
    }

    namedWindow( "Result" , WINDOW_AUTOSIZE);
    imshow( "Result", res);

    waitKey(0);
    return 0;


}