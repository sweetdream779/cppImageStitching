#include "homographyManager.h"

std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> matched){
    std::vector<cv::Point2f> res;
    for(int i=0;i<matched.size();i++){
        res.push_back(Point2f(matched[i].pt.x,matched[i].pt.y));
    } 
    return res;
}

cv::Mat HomographyManager::findOneHomo(std::vector<KeyPoint>& inliers1, std::vector<KeyPoint>& inliers2, 
									   std::vector<KeyPoint>& outliers1, std::vector<KeyPoint>& outliers2)
{

    int matchesi = (int)m_matched1.size();
    std::cout<<std::endl<<"matchesi: "<<matchesi<<std::endl;
    Mat inlier_mask, homography;
    if(m_matched1.size() >= 4) {
        homography = findHomography(Points(m_matched1), Points(m_matched2),
                                    RANSAC, ransac_thresh, inlier_mask);
        m_H_inv  = homography.inv();
    }
    if(m_matched1.size() < 4 || homography.empty()) {
        return homography;
    }

    std::vector<DMatch> inlier_matches, outlier_matches;
    int j = 0;
    int new_i;
    for(unsigned i = 0; i < m_matched1.size(); i++) {
        //only process the match if the keypoint was successfully matched
        if(inlier_mask.at<uchar>(i)) {
            new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(m_matched1[i]);
            inliers2.push_back(m_matched2[i]);
            inlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
        else
        {
           new_i = static_cast<int>(outliers1.size());
           outliers1.push_back(m_matched1[i]);
           outliers2.push_back(m_matched2[i]);
           outlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    int inliers = (int)inliers1.size();
    std::cout<<"Inliers: "<<inliers1.size()<<std::endl;
    std::cout<<"Outliers: "<<outliers1.size()<<std::endl;
    double ratio = inliers * 1.0 / matchesi;

    return homography;
}


cv::Mat HomographyManager::findMainHomo(int k)
{
    for(int i = 0; i < k; ++i)
    {

    }
}

void HomographyManager::findSeveralHomo(const int num, std::vector<cv::KeyPoint> matched1, std::vector<cv::KeyPoint> matched2)
{
    m_homographySet.resize(num);
    setMatchedPoints(matched1, matched2);
    for(int i = 0; i < num; ++i)
    {
        std::vector<KeyPoint> inliers1, inliers2;
        std::vector<KeyPoint> outliers1, outliers2;
        cv::Mat homo = findOneHomo(inliers1, inliers2, outliers1, outliers2);

        m_homographySet[i] = homo;

        setMatchedPoints(outliers1, outliers2);
    }
}

void HomographyManager::removeWrongHomo()
{
    //check if not save orintation
}

cv::Mat HomographyManager::getOptimalHomo()
{

}
