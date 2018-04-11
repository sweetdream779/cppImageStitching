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
    //TODO speed up by replace KeyPoint => cv::Point2f 
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
    setMatchedPoints(matched1, matched2);
    for(int i = 0; i < num; ++i)
    {
        std::vector<KeyPoint> inliers1, inliers2;
        std::vector<KeyPoint> outliers1, outliers2;
        cv::Mat homo = findOneHomo(inliers1, inliers2, outliers1, outliers2);
        if(homo.empty())
            continue;
        m_homographySet.push_back(homo);
        m_invHomographySet.push_back(m_H_inv);

        setMatchedPoints(outliers1, outliers2);
    }
}

std::vector<cv::Point2f> HomographyManager::transformPoints(std::vector<cv::Point2f>& pts_in, cv::Mat& homo)
{
    std::vector<cv::Point2f> pts_out(pts_in.size());
    cv::perspectiveTransform(pts_in, pts_out, homo);
    return pts_out;
}

//check if a homography not saves orientation
void HomographyManager::removeWrongHomo()
{
    std::cout<<"Num of homography before wrongs deleting: "<<m_homographySet.size()<<std::endl;
    
    int i = 0;
    while(i < m_homographySet.size())
    {
        std::vector<cv::Point2f> pts_src(2);
        pts_src[0] = cv::Point2f(0,100);
        pts_src[1] = cv::Point2f(100,200);
        std::vector<cv::Point2f> pts_target = transformPoints(pts_src, m_homographySet[i]);
        if(pts_target[0].x > pts_target[1].x || pts_target[0].y > pts_target[1].y)
        {
            m_homographySet.erase(m_homographySet.begin() + i);
            m_invHomographySet.erase(m_invHomographySet.begin() + i);            
        }
        else
            ++i;
    }

    std::cout<<"Num of homography after wrongs deleting: "<<m_homographySet.size()<<std::endl;

}

void HomographyManager::setTransformedImgs(cv::Mat& imTgt, cv::Size sizeSrc)
{   
    int num = m_homographySet.size();
    m_transformedImgs.resize(num);
    for(int i = 0; i < num; ++i)
    {
        cv::warpPerspective(imTgt, m_transformedImgs[i], m_invHomographySet[i], sizeSrc, cv::WARP_INVERSE_MAP);
    }
}
