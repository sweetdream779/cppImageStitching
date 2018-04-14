#include "myStitcher.h"
#include "reconstructer.h"

cv::Mat getSegmentation(const cv::Mat& src){
    cv::Mat seg = cv::imread("/home/irina/Desktop/im3_seg.png", 0);
    return seg;
}

std::vector<cv::Point2f> findPointsOnBoundary(cv::Mat& img, const cv::Rect& r, const cv::Mat& src)
{
    cv::Mat binary;
    cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY);

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);
    std::vector<cv::Point2f> points;

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }
            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);       

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                    int *row2 = (int*)label_image.ptr(i);
                    for(int j=rect.x; j < (rect.x+rect.width); j++) {
                        if(row2[j] != label_count) {
                            continue;
                        }
                        if(j + r.x < src.cols && i + r.y < src.rows)
                            points.push_back(cv::Point2f(j + r.x , i + r.y));
                    }
            }              

            label_count++;
        }
    }
    return points;
}

std::vector<cv::Point2f> getBoundary(const cv::Mat& croppedimg, const cv::Rect& r, const cv::Mat& src)
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int thresh = 100;
    int boundsize = 5;

    cv::Mat cropped_gray = croppedimg.clone();
    cv::Mat new_cropped_gray = Mat::zeros( cv::Size(cropped_gray.cols + 6, cropped_gray.rows + 6), CV_8UC1 );
    cv::Mat roi1(new_cropped_gray, Rect(2, 2,  cropped_gray.cols, cropped_gray.rows));
    cropped_gray.copyTo(roi1);

    //erode cropped image
    int erosion_size = 1;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
    
    cv::erode( new_cropped_gray, new_cropped_gray, element);

    Canny( new_cropped_gray, canny_output, thresh, thresh*2, 3 );
    findContours( canny_output, contours, hierarchy,  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point() );

    //Mat canvas = Mat::zeros( new_cropped_gray.size(), CV_8UC1 );
    Mat canvas;
    std::cout<<"contours: "<<contours.size()<<std::endl;
    //always contours.size() = 1
    for( int i = 0; i < contours.size(); i++ ){
        canvas = Mat::zeros( new_cropped_gray.size(), CV_8UC1 );
        drawContours( canvas, contours, i, cv::Scalar( 255,255,255 ), boundsize, 8, hierarchy, 0, Point() );
    }
    std::vector<cv::Point2f> points = findPointsOnBoundary(canvas, r, src);

    std::cout<<"contours: "<<points.size()<<std::endl;
    return points;
}

void FindBlobs(const cv::Mat &img, std::vector <DataForMinimizer>& datas, int borderX)
{
    datas.clear();
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
                //std::vector <cv::Point2f> maskPoints;

                std::vector <bool> needTransforms;
                DataForMinimizer data;
                int numPix = 0;

                for(int i=rect.y; i < (rect.y+rect.height); i++) {
                    int *row2 = (int*)label_image.ptr(i);
                    for(int j=rect.x; j < (rect.x+rect.width); j++) {
                        blob.push_back(cv::Point2f(j,i));

                        if(row2[j] != label_count) {
                            needTransforms.push_back(false);
                            //maskPoints.push_back(cv::Point2f(j,i));
                            continue;
                        }
                        needTransforms.push_back(true);
                        numPix++;
                    }
                }

                if(numPix > 10){
                    data.points = blob;
                    data.rect = rect;
                    data.needTransforms = needTransforms;
                    //data.maskPoints = maskPoints;
                    std::cout<<"rect y, x : "<<rect.y+rect.height<<" "<<rect.x+rect.width/2<<std::endl;
                    cv::Mat croppedimg = img(rect);
                    data.maskPoints = getBoundary(croppedimg, rect, img);
                    //data.rect = cv::boundingRect(data.maskPoints);
                    datas.push_back(data);
                }
            }

            label_count++;
        }
    }
}

int main(int argc, char **argv){
    Mat image1,image2,res,vis;

	image1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    image2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); 

    image1 = resizeImg(image1);
    image2 = resizeImg(image2);

    //Ptr<KAZE> akaze = KAZE::create();
    //akaze->setThreshold(akaze_thresh);

    //Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
    //detector->setThreshold(100);
    
    //Ptr<GFTTDetector> detector=GFTTDetector::create();

    Ptr<SURF> detector = SURF::create( 400 );
    Ptr<BriefDescriptorExtractor> extractor = BriefDescriptorExtractor::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    cv::Mat seg1 = cv::imread("/home/irina/Desktop/mask1.png", 0);
    cv::Mat seg2 = cv::imread("/home/irina/Desktop/mask2.png", 0);

    seg1 = resizeImg(seg1);
    seg2 = resizeImg(seg2);

    int dilation_size = 3;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
    
    cv::dilate( seg1, seg1, element);
    cv::dilate( seg2, seg2, element);

    int borderX1 = image1.cols - 1;

    int borderX2;
    std::vector<cv::KeyPoint> matched1, matched2;
    MyStitcher stitcher(detector, matcher, extractor);
    res=stitcher.stitch(image1, image2, matched1, matched2, borderX2);

    namedWindow( "Result" , WINDOW_AUTOSIZE);
    imshow( "Result", res);
    //find connected components on ipm
    //std::vector < std::vector<cv::Point2f > > blobs1;
    std::vector <DataForMinimizer> data1;
    FindBlobs(seg1, data1, borderX1);
    std::cout<<"Blobs on joint from image1: "<<data1.size()<<std::endl;


    //std::vector < std::vector<cv::Point2f> > blobs2;
    std::vector <DataForMinimizer> data2;
    FindBlobs(seg2, data2, borderX2);
    std::cout<<"Blobs on joint from image2: "<<data2.size()<<std::endl;

    Reconstructer reconstructer; 
    //pass first homo too

    //remove blob
    if(data1.size() > 0 /*&& data2.size() == 0*/)
    {
        reconstructer.reconstruct(image1, image2, cv::Size(image2.cols, image2.rows), cv::Size(image1.cols + image2.cols, image2.rows), 
                                data1, matched1, matched2);
    }
    //cv::Mat rec;
    //hconcat(image1, image2, rec);                 
    //imshow( "Reconstructed", rec);

    waitKey(0);
    return 0;
}