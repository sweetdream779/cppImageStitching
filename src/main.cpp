#include "myStitcher.h"
#include "reconstructer.h"

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
                        if(j + r.x -3 < src.cols && i + r.y -3 < src.rows)
                            points.push_back(cv::Point2f(j + r.x - 3 , i + r.y - 3));
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
    cv::Mat roi1(new_cropped_gray, Rect(3, 3,  cropped_gray.cols, cropped_gray.rows));
    cropped_gray.copyTo(roi1);

    //erode cropped image
    int erosion_size = boundsize-2;
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

void changeRect(cv::Rect& rect, const cv::Mat& img)
{
    if(rect.x - 1 > 0)
    {
        rect.x-=1;
        rect.width+=1;
    }
    if(rect.y -1 > 0){
        rect.y-=1;
        rect.height +=1;
    }
    if(rect.x + rect.width + 1 < img.cols){
        rect.width+=1;
    }
    if(rect.y + rect.height + 1 < img.rows)
        rect.height+=1;
}

int FindBlobs(const cv::Mat &img, std::vector <DataForMinimizer>& datas, int borderX, bool use_gdf, bool rigth_im = false)
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
    int onBorder = 0;
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
            changeRect(rect, img);
            DataForMinimizer data;
            data.onBorder = false;
            if(rect.x <= borderX && rect.x+rect.width >= borderX)
            {
                std::vector <cv::Point2f> blob;         

                std::vector <bool> needTransforms;
                int numPix = 0;

                for(int i=rect.y; i < (rect.y+rect.height); i++) {
                    int *row2 = (int*)label_image.ptr(i);
                    for(int j=rect.x; j < (rect.x+rect.width); j++) {

                        if(rigth_im && j < borderX)
                        {
                            if(row2[j] != label_count)
                                continue;
                            //store point for right frame
                            blob.push_back(cv::Point2f(j,i));
                            numPix++;
                        }
                        else
                        {
                            //store point for left frame
                            blob.push_back(cv::Point2f(j,i));

                            if(row2[j] != label_count) {
                                needTransforms.push_back(false);
                                continue;
                            }
                            needTransforms.push_back(true);
                            numPix++;
                        }
                    }
                }

                if(numPix > 15){
                    data.points = blob;
                    data.needTransforms = needTransforms;
                    if(use_gdf && rigth_im == false)
                    {
                        //additional data for gradient domain fusion
                        cv::Mat croppedimg = img(rect);
                        data.maskPoints = getBoundary(croppedimg, rect, img);
                    }
                    data.onBorder = true;
                    onBorder++;
                    data.rect = rect;
                    datas.push_back(data);
                }
            }
            if(rect.width > 5 && rect.height > 5 && !data.onBorder)
            {
                data.rect = rect;
                datas.push_back(data);
            }

            label_count++;
        }
    }
    return onBorder;
}

cv::Mat resizeMask(const cv::Mat& im, cv::Size size)
{
    cv::Mat  img = im.clone();
    cv::resize(img, img, size);
    return img;
}

cv::Mat process(cv::Mat image1, cv::Mat image2, cv::Mat seg1, cv::Mat seg2, bool use_gdf, bool repairSeam)
{
    cv::Mat res, vis;
    image1 = resizeImg(image1);
    image2 = resizeImg(image2);

    //Ptr<KAZE> akaze = KAZE::create();
    //akaze->setThreshold(akaze_thresh);

    //Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
    //detector->setThreshold(100);
    
    //Ptr<GFTTDetector> detector=GFTTDetector::create();

    Ptr<SURF> detector = SURF::create();
    //Ptr<BriefDescriptorExtractor> extractor = BriefDescriptorExtractor::create();
    Ptr<SurfDescriptorExtractor> extractor = SurfDescriptorExtractor::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    seg1 = resizeMask(seg1, image1.size());
    seg2 = resizeMask(seg2, image2.size());

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
    res = stitcher.stitch(image1, image2, matched1, matched2, borderX2);
    cv::Mat homo = stitcher.getMainHomo();
    cv::Mat invHomo = stitcher.getMainInvHomo();
    if(homo.empty())
    {
        std::cout<<"Stitching failed"<<std::endl;
        return res;
    }
    //find connected components on ipm
    std::vector <DataForMinimizer> data1;
    int onBorder1 = FindBlobs(seg1, data1, borderX1, use_gdf, false);
    std::cout<<"Blobs on joint from image1: "<<onBorder1<<" ,all: "<< data1.size()<<std::endl;

    std::vector <DataForMinimizer> data2;
    cv::Mat transformedSeg2;
    cv::warpPerspective(seg2, transformedSeg2, homo, cv::Size(image1.cols + image2.cols, image1.rows));
    int onBorder2 = FindBlobs(transformedSeg2, data2, borderX1, false, true);
    std::cout<<"Blobs on joint from image2: "<<onBorder2<<" ,all: "<<data2.size()<<std::endl;

    int homoNum = 2;
    Reconstructer reconstructer(use_gdf, homoNum); 
    
    //remove blob
    if(onBorder1 > 0 && onBorder2 == 0)
    {
        std::cout<<"First reconstruction"<<std::endl;
        image1 = reconstructer.reconstructWithRemoval(image1, image2, cv::Size(image1.cols + image2.cols, image2.rows), 
                                                      data1, data2, matched1, matched2);

    }
    if(onBorder2 > 0 && onBorder1 == 0)
    {
        std::cout<<"Second reconstruction"<<std::endl;
        image1 = reconstructer.reconstructWithAdding(homo, invHomo, data2, data1, image1, image2, res, borderX1);
    }
    if(repairSeam){}
    res = stitcher.fill_and_crop(res, image1, image2);
}

int main(int argc, char **argv){
    bool use_gdf = true; //use gradient domain fusion in reconstruction or not
    bool repairSeam = true;

    Mat image1,image2,res;

	image1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    image2 = imread(argv[2], CV_LOAD_IMAGE_COLOR); 

    cv::Mat seg1 = cv::imread("/home/irina/ICNet-tensorflow/output/left.png", 0);
    cv::Mat seg2 = cv::imread("/home/irina/ICNet-tensorflow/output/right2.png", 0);
    
    res = process(image1, image2, seg1, seg2, use_gdf, repairSeam); 
    cv::imwrite("/home/irina/Desktop/diploma_result/removingResult3.jpg",res);     
    cv::imshow( "Reconstructed", res);

    waitKey(0);
    
    std::cout<<"Finished\n";
    return 0;
}