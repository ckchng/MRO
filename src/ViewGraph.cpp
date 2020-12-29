

#include "ViewGraph.hpp"
#include "ViewDatabase.hpp"
#include <h5cpp/hdf5.hpp>
#include <ctime>
#include <cmath>

#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>

#define HISTO_LENGTH 30
#define TH_LOW 50

#define PLOT true

using namespace irotavg;
using namespace hdf5;
using namespace opengv;

bool ViewGraph::checkDistEpipolarLine(const cv::KeyPoint &kp1,
                                           const cv::KeyPoint &kp2,
                                           const cv::Mat &F12) const
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    //TODO: use matrix operations
    // vector coefs = kp1.pt.x.transpose() * F12

    const double a = kp1.pt.x*F12.at<double>(0,0) + kp1.pt.y*F12.at<double>(1,0) + F12.at<double>(2,0);
    const double b = kp1.pt.x*F12.at<double>(0,1) + kp1.pt.y*F12.at<double>(1,1) + F12.at<double>(2,1);
    const double c = kp1.pt.x*F12.at<double>(0,2) + kp1.pt.y*F12.at<double>(1,2) + F12.at<double>(2,2);

    const float num = a*kp2.pt.x + b*kp2.pt.y + c;

    const float den = a*a + b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;
    return dsqr<3.84*m_scale_sigma_squares[kp2.octave];
}


void computeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0, max2=0, max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = (int)histo[i].size();
        if(s>max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if(s>max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2 < 0.1f*(float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if(max3 < 0.1f*(float)max1)
    {
        ind3 = -1;
    }
}


int descriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


int ViewGraph::findORBMatchesByBoW(Frame &f1, Frame &f2,
                                        std::vector<std::pair<int,int> > &matched_pairs,
                                        const double nnratio) const
{
    bool check_orientation = true;

    const auto &keypoints1 = f1.undistortedKeypoints();
    const auto &keypoints2 = f2.undistortedKeypoints();
    const auto &descriptors1 = f1.descriptors();
    const auto &descriptors2 = f2.descriptors();
    const auto &bow_features1 = f1.bow_features();
    const auto &bow_features2 = f2.bow_features();

    const int n1 = (int)keypoints1.size();
    const int n2 = (int)keypoints2.size();


    int n_matches = 0;

    //vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
    std::vector<int> matches12(n1,-1);
    std::vector<int> matched2(n2,-1);



    std::vector<int> rotHist[HISTO_LENGTH];

    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(5000); //500

    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)


    DBoW2::FeatureVector::const_iterator it1 = bow_features1.begin();
    DBoW2::FeatureVector::const_iterator it2 = bow_features2.begin();
    DBoW2::FeatureVector::const_iterator end1 = bow_features1.end();
    DBoW2::FeatureVector::const_iterator end2 = bow_features2.end();

    while(it1!=end1 && it2!=end2)
    {
        if(it1->first == it2->first)
        {


            //for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            for(size_t i1=0, iend1=it1->second.size(); i1<iend1; i1++)
            {
                //const unsigned int realIdxKF = vIndicesKF[iKF];
                const int idx1 = it1->second[i1];
                const cv::KeyPoint &kp1 = keypoints1[idx1];
                const cv::Mat &d1 = descriptors1.row(idx1);


                int best_dist1 = 256;
                int best_idx2  = -1 ;
                int best_dist2 = 256;

                for(size_t i2=0, iend2=it2->second.size(); i2<iend2; i2++)
                {

                    const int idx2 = it2->second[i2];

                    if(matched2[idx2] > -1){
                        continue;
                    }


                    const cv::Mat &d2 = descriptors2.row(idx2);
                    const int dist = descriptorDistance(d1, d2);


                    if(dist < best_dist1)
                    {
                        best_dist2 = best_dist1;
                        best_dist1 = dist;
                        best_idx2 = idx2;
                    }
                    else if(dist < best_dist2)
                    {
                        best_dist2 = dist;
                    }
                }

                if(best_dist1 <= TH_LOW)
                {
                    const cv::KeyPoint &kp2 = keypoints2[best_idx2];

                    if( best_dist1 < nnratio*best_dist2 )
                    {

                        matches12[idx1] = best_idx2;
                        matched2[best_idx2] = 2;


                        if(check_orientation)
                        {
                            double rot = kp1.angle - kp2.angle;
                            if(rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot*factor);
                            if(bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);

                        }
                        n_matches++;
                    }
                }
            }

            it1++;
            it2++;
        }
        else if(it1->first < it2->first)
        {
            it1 = bow_features1.lower_bound(it2->first);
        }
        else
        {
            it2 = bow_features2.lower_bound(it1->first);
        }
    }


    if(check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        computeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                matches12[rotHist[i][j]]=-1;
                n_matches--;
            }
        }
    }

    matched_pairs.clear();
    matched_pairs.reserve(n_matches);

    for(size_t i=0, iend=matches12.size(); i<iend; i++)
    {
        if(matches12[i]<0)
            continue;
        matched_pairs.push_back(make_pair(i,matches12[i]));
    }

    return n_matches;
}


int ViewGraph::findORBMatches(Frame &f1, Frame &f2,
                                   cv::Mat F12,
                                   std::vector<std::pair<int,int> > &matched_pairs) const
{
    bool check_orientation = true;

    const auto &keypoints1 = f1.undistortedKeypoints();
    const auto &keypoints2 = f2.undistortedKeypoints();
    const auto &descriptors1 = f1.descriptors();
    const auto &descriptors2 = f2.descriptors();
    const auto &bow_features1 = f1.bow_features();
    const auto &bow_features2 = f2.bow_features();

    const int n1 = (int)keypoints1.size();


    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int n_matches = 0;
    std::vector<int> matches12(n1,-1);

    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0; i<HISTO_LENGTH; i++)
        rotHist[i].reserve(5000);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator it1 = bow_features1.begin();
    DBoW2::FeatureVector::const_iterator it2 = bow_features2.begin();
    DBoW2::FeatureVector::const_iterator end1 = bow_features1.end();
    DBoW2::FeatureVector::const_iterator end2 = bow_features2.end();

    while(it1!=end1 && it2!=end2)
    {
        if(it1->first == it2->first)
        {
            for(size_t i1=0, iend1=it1->second.size(); i1<iend1; i1++)
            {
                const int idx1 = it1->second[i1];
                const cv::KeyPoint &kp1 = keypoints1[idx1];
                const cv::Mat &d1 = descriptors1.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;

                for(size_t i2=0, iend2=it2->second.size(); i2<iend2; i2++)
                {
                    const int idx2 = it2->second[i2];

                    // If we have already matched or there is a MapPoint skip

                    if (matches12[idx2]>-1)
                    {
                        continue;
                    }

                    const cv::Mat &d2 = descriptors2.row(idx2); // I think & really make any difference here...
                    const int dist = descriptorDistance(d1,d2);

                    if(dist > TH_LOW || dist > bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = keypoints2[idx2];

                    if(checkDistEpipolarLine(kp2,kp1,F12))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if(bestIdx2 >= 0)
                {
                    const cv::KeyPoint &kp2 = keypoints2[bestIdx2];
                    matches12[idx1] = bestIdx2;
                    n_matches++;

                    if(check_orientation)
                    {
                        double rot = kp1.angle - kp2.angle;
                        if(rot < 0.0)
                            rot += 360.0f;
                        int bin = round(rot*factor);
                        if(bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }
            it1++;
            it2++;
        }
        else if(it1->first < it2->first)
        {
            it1 = bow_features1.lower_bound(it2->first);
        }
        else
        {
            it2 = bow_features2.lower_bound(it1->first);
        }
    }

    if(check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        computeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                matches12[rotHist[i][j]]=-1;
                n_matches--;
            }
        }
    }

    matched_pairs.clear();
    matched_pairs.reserve(n_matches);

    for(size_t i=0, iend=matches12.size(); i<iend; i++)
    {
        if(matches12[i]<0)
            continue;
        matched_pairs.push_back(make_pair(i,matches12[i]));
    }

    return n_matches;
}


int findORBMatchesLocally(Frame &f1, Frame &f2,
                          const std::vector<cv::Point2f> &guess_matching_pts,
                          std::vector<int> &matches12,
                          const int window_size, const double nnratio)
{
    bool const check_orientation = false;
    int n_matches = 0;

    auto &points1 = f1.undistortedKeypoints();
    auto &points2 = f2.undistortedKeypoints();

    auto &descriptors1 = f1.descriptors();
    auto &descriptors2 = f2.descriptors();

    const size_t n1 = points1.size();
    const size_t n2 = points2.size();

    assert(n1==guess_matching_pts.size());

    matches12 = std::vector<int>(n1,-1);

    std::vector<int> rot_hist[HISTO_LENGTH];
    for(int i=0; i<HISTO_LENGTH; i++)
        rot_hist[i].reserve(5000);

    const float factor = 1.0f/HISTO_LENGTH;

    std::vector<int> matched_distances(n2,INT_MAX);
    std::vector<int> matches21(n2,-1);

    for(int i1=0; i1<n1; i1++)
    {
        const cv::KeyPoint &kp1 = points1[i1];

        int level = kp1.octave;
        int min_level = MAX(0,level-2);
        int max_level = MIN(level+2,7);

        std::vector<int> indices2 = f2.getFeaturesInArea(guess_matching_pts[i1].x,
                                                         guess_matching_pts[i1].y,
                                                         window_size,
                                                         min_level,max_level);
        if(indices2.empty())
            continue;

        cv::Mat d1 = descriptors1.row(i1);

        int best_dist = INT_MAX;
        int best_dist2 = INT_MAX;
        int best_idx2 = -1;

        for (int i2: indices2)
        {
            cv::Mat d2 = descriptors2.row(i2);
            int dist = descriptorDistance(d1,d2);

            if (matched_distances[i2]<=dist)
                continue;

            if (dist < best_dist)
            {
                best_dist2 = best_dist;
                best_dist = dist;
                best_idx2 = i2;
            }
            else if (dist < best_dist2)
            {
                best_dist2 = dist;
            }
        }

        if (best_dist <= TH_LOW)
        {
            if (best_dist < (float)best_dist2*nnratio)
            {
                if (matches21[best_idx2] >= 0) // (i,j) //j has been assigned
                {
                    matches12[matches21[best_idx2]] = -1;
                    n_matches--;
                }

                matches12[i1] = best_idx2;
                matches21[best_idx2] = i1;
                matched_distances[best_idx2] = best_dist;
                n_matches++;

                if(check_orientation)
                {
                    float rot = points1[i1].angle - points2[best_idx2].angle;
                    if(rot<0.0)
                        rot += 360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rot_hist[bin].push_back(i1);
                }
            }
        }
    }

    if(check_orientation)
    {
        int ind1 = -1, ind2 = -1, ind3 = -1;

        computeThreeMaxima(rot_hist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rot_hist[i].size(); j<jend; j++)
            {
                int idx1 = rot_hist[i][j];
                if(matches12[idx1]>=0)
                {
                    matches12[idx1]=-1;
                    n_matches--;
                }
            }
        }
    }

    return n_matches;
}




int findCurr2PrevLocalMatches(Frame &curr, Frame &prev,
                              std::vector<int> &target, int img_width, int img_height, const int rad = 100)
{
    const double nnratio = .9; //1.0  ; //.8 .9;

    auto &curr_points = curr.undistortedKeypoints();
    const int n_curr_points = (int)curr_points.size();
    std::vector<cv::Point2f> guess_matching_points;
    guess_matching_points.reserve(n_curr_points);

    for(int i=0; i<n_curr_points; i++)
    {
        guess_matching_points.push_back(curr_points[i].pt);
    }

    // change to current_to_pivot_map
    //std::vector<int> target; //-1 if could not find any match or untracked

    int found_matches = findORBMatchesLocally(curr, prev, guess_matching_points,
                                               target, rad, nnratio);

    // heuristic to ensure the matches are evenly distributed
    // first find the mean number of each segment should have
    int mean_num_points = 0;
    mean_num_points = found_matches / 50;

    std::vector<int> segment(50, 0);
    double curr_x = 0;
    double curr_y = 0;
    int row_id = 0;
    int col_id = 0;
    int num_col = 10;
    int img_height_sampling = img_height / 5;
    int img_width_sampling = img_width / 10;
    int num_of_remaining_features = 0;


    for (int i = 0; i < target.size(); i++) {
        if (target[i] > -1) {
            // obtain x and y
            curr_x = guess_matching_points[i].x;
            curr_y = guess_matching_points[i].y;

            col_id = ceil(curr_x / img_width_sampling);
            row_id = ceil(curr_y / img_height_sampling);

            // then decide the segment
            int seg_id = 0;
            seg_id = (((row_id - 1) * num_col) + col_id) - 1;
            assert(seg_id >= 0 && seg_id < 50);
            segment[seg_id] = segment[seg_id] + 1;

            // throw away the matches if the number of segment has exceeded the mean number
            if (segment[seg_id] > mean_num_points) {
                target[i] = -1;
            } else {
                num_of_remaining_features = num_of_remaining_features + 1;
            }

        }
    }


//    std::cout << "started with: " << found_matches << std::endl;
//    std::cout << "ended with: " << num_of_remaining_features << std::endl;
    return num_of_remaining_features;

}



Pose ViewGraph::findRelativePose_with_eigsolver(Frame &f1, Frame &f2,
                                        std::vector<double> prevRrel, int &ransac_iter,
                                      const int MIN_SET, const int max_rsc_iter,
                                      FeatureMatches &matches,
                                      std::vector<int> &inlier_vec,
                                      int &n_cheirality, cv::Mat &mask, cv::Mat &E,
                                      double th) const
{

    assert(matches.size()>10);

    const auto &kps1 = f1.undistortedKeypoints();
    const auto &kps2 = f2.undistortedKeypoints();

    std::vector<cv::Point2d> points1, points2;
    const int n_matches = (int)matches.size();
    points1.reserve(n_matches);
    points2.reserve(n_matches);

    for (auto &match: matches)
    {
        assert(match.queryIdx>=0 && match.trainIdx>=0);
        points1.push_back( kps1[match.queryIdx].pt );
        points2.push_back( kps2[match.trainIdx].pt );
    }

    const Camera &cam = Camera::instance();
    const CameraParameters &cam_pars = cam.cameraParameters();

    const cv::Mat K = cv::Mat(cam.cameraParameters().intrinsic());
    const cv::Mat K_inv = K.inv();
    const cv::Mat K_inv_t = K_inv.t();

    const double focal = cam_pars.f();
    const cv::Point2d pp = cam_pars.pp();

    // first, convert measurements to F1 and F2
    bearingVectors_t bearingVectors1;
    bearingVectors_t bearingVectors2;

//    std::vector<double> tmp_f1;
    cv::Vec3d tmp_f1;
    cv::Vec3d tmp_f2;
    translation_t bv1;
    translation_t bv2;

    cv::Mat tmp_f1_K_inv(3, 1, CV_64F);
    cv::Mat tmp_f2_K_inv(3, 1, CV_64F);

    for (int p_id = 0; p_id < points1.size(); p_id++){
        // first get points1 and points2
       tmp_f1[0] = points1[p_id].x;
       tmp_f1[1] = points1[p_id].y;
       tmp_f1[2] = 1.0;

        tmp_f2[0] = points2[p_id].x;
        tmp_f2[1] = points2[p_id].y;
        tmp_f2[2] = 1.0;

       tmp_f1_K_inv = K_inv * cv::Mat(tmp_f1);
       tmp_f2_K_inv = K_inv * cv::Mat(tmp_f2);
       double tmp_f1_norm = 0;
       double tmp_f2_norm = 0;
       for(int i = 0; i< 3; i++){
           tmp_f1_norm =  tmp_f1_norm + (tmp_f1_K_inv.at<double>(i, 0) * tmp_f1_K_inv.at<double>(i, 0));
           tmp_f2_norm =  tmp_f2_norm + (tmp_f2_K_inv.at<double>(i, 0) * tmp_f2_K_inv.at<double>(i, 0));
       }

       tmp_f1_norm = sqrt(tmp_f1_norm);
       tmp_f2_norm = sqrt(tmp_f2_norm);

        for(int i = 0; i< 3; i++){
           bv1[i] =  tmp_f1_K_inv.at<double>(i, 0) / tmp_f1_norm;
           bv2[i] =  tmp_f2_K_inv.at<double>(i, 0) / tmp_f2_norm;
        }

        bearingVectors1.push_back(bv1);
        bearingVectors2.push_back(bv2);

    }


    // initialise with previous Rotation
    rotation_t rotation;
    for (int i = 0; i<9; i++){
        rotation(i) = prevRrel[i];
    }

    relative_pose::CentralRelativeAdapter adapter(
            bearingVectors1,
            bearingVectors2,
            rotation);


    sac::Ransac<
    sac_problems::relative_pose::EigensolverSacProblem> ransac;
    std::shared_ptr<
            sac_problems::relative_pose::EigensolverSacProblem> eigenproblem_ptr(
            new sac_problems::relative_pose::EigensolverSacProblem(adapter, MIN_SET));
    ransac.sac_model_ = eigenproblem_ptr;
    ransac.threshold_ = 1e-6;
    ransac.max_iterations_ = max_rsc_iter;

    ransac.computeModel();

    ransac_iter = ransac.iterations_;

//    std::cout<<"Ransac needed "<<ransac.iterations_<< "iterations."<<std::endl;
    sac_problems::relative_pose::EigensolverSacProblem::model_t optimizedModel;
    eigenproblem_ptr->optimizeModelCoefficients(
            ransac.inliers_,
            ransac.model_coefficients_,
            optimizedModel);

    std::vector<int> inlier_out(ransac.inliers_.size());
    for (size_t i = 0; i < ransac.inliers_.size(); i++) {
        inlier_out[i] = ransac.inliers_[i];
    }

    inlier_vec.insert(inlier_vec.end(), inlier_out.begin(), inlier_out.end());


    cv::Mat R(3, 3, CV_64F);
    cv::Mat t(3, 1, CV_64F);
    std::vector<double> R_vec(9);
    rotation_t tmp_R;
    translation_t tmp_t;
    tmp_R = optimizedModel.rotation;
    tmp_t = optimizedModel.translation_norm;

    n_cheirality = ransac.inliers_.size();


    for(int r_id = 0; r_id<3; r_id++){
        for(int c_id = 0; c_id<3; c_id++){
            R.at<double>(r_id, c_id) = optimizedModel.rotation((r_id*3) + c_id);
        }
    }

    for(int t_id = 0; t_id<3; t_id++){
        t.at<double>(t_id, 0) = optimizedModel.translation_norm(t_id);
    }


    return Pose(R, t);

}


Pose ViewGraph::findRelativePose(Frame &f1, Frame &f2,
                                 FeatureMatches &matches,
                                 int &n_cheirality, cv::Mat &mask, cv::Mat &E,
                                 double th) const
{
    //https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    assert(matches.size()>4);

    const auto &kps1 = f1.undistortedKeypoints();
    const auto &kps2 = f2.undistortedKeypoints();

    std::vector<cv::Point2d> points1, points2;
    const int n_matches = (int)matches.size();
    points1.reserve(n_matches);
    points2.reserve(n_matches);

    for (auto &match: matches)
    {
        assert(match.queryIdx>=0 && match.trainIdx>=0);
        points1.push_back( kps1[match.queryIdx].pt );
        points2.push_back( kps2[match.trainIdx].pt );
    }

    const Camera &cam = Camera::instance();
    const CameraParameters &cam_pars = cam.cameraParameters();

    const double focal = cam_pars.f();
    const cv::Point2d pp = cam_pars.pp();

    E = cv::findEssentialMat(points1, points2,
                             focal, pp, cv::RANSAC, 0.999, th, mask);

    int n_rsc_inlrs = 0;
    const uchar* mask_ptr = mask.ptr<uchar>(0);
    for (int i=0; i<mask.rows; i++)
        n_rsc_inlrs += mask_ptr[i];

    if (n_rsc_inlrs>6) // is 5 the min num of pts ?
    {
        cv::Mat R, t;
        //int n_cheirality;
        //cv::Mat mask_pose = mask.clone();
        n_cheirality = cv::recoverPose(E, points1, points2, R, t, focal, pp, mask);
        return Pose(R,t);
    }
    else
    {
        n_cheirality = 0;
        return Pose();
    }
}

void plotMatches(Frame &prev_frame, Frame &curr_frame, FeatureMatches &matches)
{
    auto im1 = prev_frame.getImage();
    auto im2 = curr_frame.getImage();
    auto kps1 = prev_frame.keypoints();
    auto kps2 = curr_frame.keypoints();

    cv::Mat im_matches;
    cv::drawMatches(im1, kps1, im2, kps2, matches, im_matches);
    double s=1.0f;
    cv::Size size(s*im_matches.cols,s*im_matches.rows);
    resize(im_matches,im_matches,size);
    cv::imshow("matches after ransac", im_matches);
    cv::waitKey(1);
}


void ViewGraph::filterMatches(FeatureMatches &matches, const cv::Mat &inlrs_mask, int n_epi_inlrs) const
{
    if (n_epi_inlrs==0)
    {
        matches.clear();
        return;
    }

    FeatureMatches tmp_matches;
    tmp_matches.reserve(n_epi_inlrs);

    const uchar* inlrs_mask_ptr = inlrs_mask.ptr<uchar>(0);
    for(int i = 0; i <matches.size(); i++)
    {
        if (inlrs_mask_ptr[i])
        {
            tmp_matches.push_back( std::move(matches[i]) );
        }
    }
    matches = std::move(tmp_matches); // no not use epi_matches from here...
    assert(matches.size()==n_epi_inlrs);
}

void ViewGraph::filterMatches_w_inlr_id(FeatureMatches &matches, const std::vector<int>inlier_vec, int n_epi_inlrs) const
{
    if (n_epi_inlrs==0)
    {
        matches.clear();
        return;
    }

    FeatureMatches tmp_matches;
    tmp_matches.reserve(n_epi_inlrs);

    for(int i = 0; i <inlier_vec.size(); i++)
    {
        assert(matches[inlier_vec[i]].trainIdx >= 0 && matches[inlier_vec[i]].queryIdx >= 0);
        tmp_matches.push_back( std::move(matches[inlier_vec[i]]) );
     }
    matches = std::move(tmp_matches); // no not use epi_matches from here...
    assert(matches.size()==n_epi_inlrs);
}



int ViewGraph::refinePose(Frame &f1, Frame &f2, const int min_set, Pose &pose, cv::Mat &E_best, FeatureMatches &matches) const
{
    // TODO pass min_matches as parameter
    const int min_matches = 100;
    if (matches.size()<5)
    {
        return (int)matches.size();
    }

    const int max_iters = 10;
    const Camera &cam = Camera::instance();
    const cv::Mat K = cv::Mat(cam.cameraParameters().intrinsic());
    const cv::Mat K_inv = K.inv();
    const cv::Mat K_inv_t = K_inv.t();

    int best_inlrs = (int)matches.size();
    cv::Mat E = E_best;

    std::vector< std::pair<int,int> > matched_pairs;
    cv::Mat inlrs_mask;

    int inlrs;
    int iters = 1;
    do // use essential matrix in this loop
    {
        cv::Mat F = K_inv_t*E_best*K_inv;

        matched_pairs.clear();
        if (findORBMatches(f1, f2, F, matched_pairs) <.75*min_matches)
        {
            break;
        }

        FeatureMatches curr_matches;

        for (auto &match_pair: matched_pairs)
            curr_matches.push_back( cv::DMatch(match_pair.first, match_pair.second, 0) );

        inlrs_mask = cv::Mat();
        std::vector<int>inlier_vec;

        Pose curr_pose = findRelativePose(f1, f2, matches, inlrs, inlrs_mask, E);

        if (inlrs > best_inlrs )
        {
            best_inlrs = inlrs;
            E_best = E;
            matches = std::move(curr_matches);
            pose = std::move(curr_pose);
            filterMatches(matches, inlrs_mask, inlrs); //move out of the while

        }
        else
        {
            break;
        }
    }
    while (iters++<max_iters);


    std::cout<< "refinePose iterations " << iters <<std::endl;
    return (int)matches.size();
}


bool ViewGraph::findPose(View &v1, View &v2, View &pivot, const int min_set,
                              const std::vector<int> pivot2v2, Pose &pose,
                              cv::Mat &E_out, FeatureMatches &matches)
{

    matches.clear();
    if (!v1.isConnectedTo(pivot))
    {
        return false;
    }

    matches = v1.getFeatureMatches(pivot);

    // extend the matches from pivot to v2 view : (v1, pivot) --> (v1, v2)
    FeatureMatches tmp_matches;

    tmp_matches.reserve(matches.size());

    for (auto match : matches) // prev --> pivot : (match.queryIdx, match.trainIdx)
    {
        if (pivot2v2[match.trainIdx] >= 0)
        {
            match.trainIdx = pivot2v2[match.trainIdx];
            assert(match.trainIdx>=0);
            tmp_matches.push_back(match);
        }
    }

    matches = std::move(tmp_matches);


    if (matches.size()>5)
    {
        cv::Mat inlrs_mask;
        int n_epi_inlrs;
        std::vector<int>inlier_vec;

        pose = findRelativePose(v1.frame(), v2.frame(), matches, n_epi_inlrs, inlrs_mask, E_out);

        filterMatches(matches, inlrs_mask, n_epi_inlrs);

        return true;
    }
    else
    {
        return false;
    }
}


Pose ViewGraph::findInitialPose(View &v1, View &v2, const int prev_view_idx, int &n_epi_inlrs,
                                     cv::Mat &E, FeatureMatches &matches, int min_matches, const int MIN_SET,
                                     const int img_width, const int img_height)
{
    Pose pose;

    std::vector<int> curr2prev_map; //-1 if could not find any match or untracked


    auto &f1 = v1.frame();
    auto &f2 = v2.frame();
    // obtain patch size based on the point distance!

    double rad = 100;

    int n_curr2prev_matches = 10;
    int init_match_iter = 1;


    // keep doing while matching size is bigger than 10 but smaller than a threshold
    while ((n_curr2prev_matches >= 10) && (n_curr2prev_matches < 300) && (init_match_iter < 5)) {

        curr2prev_map.clear();
        n_curr2prev_matches = findCurr2PrevLocalMatches(f2, f1,
                                                        curr2prev_map, img_width, img_height, rad);

        // find and update mean rad
        std::vector<double> dists;
        dists.reserve(f2.undistortedKeypoints().size());
        for (int k = 0; k < f2.undistortedKeypoints().size(); k++) {
            const auto &idx = curr2prev_map[k];
            if (idx >= 0) {
                const auto &p2 = f2.undistortedKeypoints()[k];
                const auto &p1 = f1.undistortedKeypoints()[idx];
                double d = cv::norm(p1.pt - p2.pt);
                dists.push_back(d);
            }
        }

        double mean_dist = std::accumulate(dists.begin(), dists.end(), 0.0) / dists.size();
        m_local_rad = mean_dist; //.5*(m_local_rad + mean_dist);

        matches.clear();
        for (int curr_idx = 0; curr_idx < curr2prev_map.size(); curr_idx++) {
            const int &prev_idx = curr2prev_map[curr_idx]; // in prev
            if (prev_idx != -1) {
                matches.push_back(cv::DMatch(prev_idx, curr_idx, 0));
            }
        }
        init_match_iter = init_match_iter + 1;

        rad = rad * 1.5;
    }


    if (n_curr2prev_matches < 300){

        n_epi_inlrs = 0;
        std::cout << "insufficient matches... " << std::endl;
        return pose;

    }

    std::vector<int> inliers_vec;
    cv::Mat inlrs_mask;


    int ransac_iter = 0;

    //access previous relPose here.
    std::vector<double> prevRrel_vec(9);
    //////
    if (f2.id() == 1) {
        prevRrel_vec[0] = 1;
        prevRrel_vec[1] = 0;
        prevRrel_vec[2] = 0;

        prevRrel_vec[3] = 0;
        prevRrel_vec[4] = 1;
        prevRrel_vec[5] = 0;

        prevRrel_vec[6] = 0;
        prevRrel_vec[7] = 0;
        prevRrel_vec[8] = 1;
    }
    //
    if (f2.id() > 1) {
        for (const auto &x : v1.connections()) {
            auto v_ptr = x.first; // ptr to connected view to j
            auto matches_ptr = x.second; // ptr to ViewConnection object
            int i = v_ptr->frame().id();

            if (i == prev_view_idx - 1) {
                // save tij here
                const auto &prevPose = matches_ptr->pose(); //get pose
                const auto &prevRrel = prevPose.R();

                for (int r_id = 0; r_id < 9; r_id++) {
                    prevRrel_vec[r_id] = prevRrel.val[r_id];
                }
                break;
            } else {
                prevRrel_vec[0] = 1;
                prevRrel_vec[1] = 0;
                prevRrel_vec[2] = 0;

                prevRrel_vec[3] = 0;
                prevRrel_vec[4] = 1;
                prevRrel_vec[5] = 0;

                prevRrel_vec[6] = 0;
                prevRrel_vec[7] = 0;
                prevRrel_vec[8] = 1;
            }
        }
    }

    pose = findRelativePose_with_eigsolver(f1, f2, prevRrel_vec, ransac_iter, MIN_SET, 5000, matches, inliers_vec,
                                           n_epi_inlrs, inlrs_mask, E);

    // dont connect if ransac needs 5k iter
    if (ransac_iter > 5000){
        n_epi_inlrs = 0;
        return pose;
    }

    if (n_epi_inlrs > 1 * min_matches) {
        filterMatches_w_inlr_id(matches, inliers_vec, n_epi_inlrs);
        return pose;
    }


}


//TODO: return the candidate view
bool ViewGraph::detectLoopCandidates(View &view, std::vector<View*> &candidates)
{
    ORBVocabulary &orb_vocab = ORBVocabulary::instance();
    auto &vocab = orb_vocab.vocabulary();

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
//    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const auto &bow1 = view.frame().bow();
    const View::Connections &connections = view.connections();

    float min_score = 1;
    float score;
    for (const auto &c: connections)
    {
        View *v2 = c.first;

        const auto &bow2 = v2->frame().bow();

        score = vocab.score(bow1, bow2);
        if(score < min_score)
            min_score = score;
    }

    // Query the database imposing the minimum score
    auto &db = ViewDatabase::instance();
    candidates = db.detectLoopCandidates(view, min_score);

    // If there are no loop candidates, just add new keyframe and return false
    if(candidates.empty())
    {
        return false;
    }
    else
    {
        return true;
    }
}



bool ViewGraph::checkLoopConsistency(const std::vector<View*> &loop_candidates,
                          std::vector<View*> &consistent_candidates,
                          std::vector<ConsistentGroup> &consistent_groups,
                          std::vector<ConsistentGroup> &prev_consistent_groups,
                          const int covisibility_consistency_th )
{
    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it

    consistent_candidates.clear(); //  mvpEnoughConsistentCandidates.clear();
    //std::vector<ConsistentGroup> consistent_groups;   // vCurrentConsistentGroups
    consistent_groups.clear();

    std::vector<bool> prev_consistent_group_flag(prev_consistent_groups.size(),false);  //vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);


    for (View *candidate : loop_candidates) //for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {

        // --- Create candidate group -----------
        std::set<View*> candidate_group;
        const auto &connections = candidate->connections();
        for(auto pair: connections)
        {
            View *connected_view = pair.first;
            candidate_group.insert(connected_view);
        }
        candidate_group.insert(candidate);


        // --- compare candidate grou against prevoius consistent groups -----------

        bool enough_consistent = false;
        bool consistent_for_some_group = false;

        for(size_t g=0, iendG=prev_consistent_groups.size(); g<iendG; g++) //for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            // find if candidate_group is consistent with any previous consistent group
            std::set<View*> prev_group = prev_consistent_groups[g].first;
            bool consistent = false;
            for (View *candidate: candidate_group)
            {
                if( prev_group.count(candidate) )
                {
                    consistent = true;
                    consistent_for_some_group = true;
                    break;
                }
            }

            if(consistent)
            {
                int previous_consistency = prev_consistent_groups[g].second;
                int current_consistency = previous_consistency + 1;

                if( !prev_consistent_group_flag[g] )
                {
                    consistent_groups.push_back(std::make_pair(candidate_group,
                                                               current_consistency));
                    prev_consistent_group_flag[g] = true; //this avoid to include the same group more than once
                }

                if(current_consistency >= covisibility_consistency_th && !enough_consistent)
                {
                    consistent_candidates.push_back(candidate);
                    enough_consistent = true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!consistent_for_some_group)
        {
            consistent_groups.push_back(std::make_pair(candidate_group,0));
        }
    }


    return !consistent_candidates.empty();
}

bool ViewGraph::processFrame(Frame &frame,
                             const int min_set,
                             const int img_width,
                             const int img_height,
                             const int win_size,
                             const int min_matches
                             )
{
    const int skip = 0;

    // Create View
    View *curr_view = new View(frame);
    Frame &curr_frame = curr_view->frame();

    const int m = 1+(int)m_views.size();

    if (m <= skip+1) // we are done for the first frames
    {
        m_views.push_back(curr_view);
        m_fixed_mask.push_back(false);

        return true;
    }

    const int curr_view_idx = m-1;
    int prev_view_idx = curr_view_idx-skip-1;

    int count_connections = 0;


    // ---------------------------------------------
    //   local search for prev frame
    // ---------------------------------------------
    View *prev_view = m_views[prev_view_idx];
    Frame &prev_frame = prev_view->frame();

    cv::Mat E;
    FeatureMatches matches;
    int n_epi_inlr;
    std::cout<<"finding init pose"<<std::endl;
    Pose relPose = findInitialPose(*prev_view, *curr_view, prev_view_idx, n_epi_inlr, E, matches, min_matches, min_set, img_width, img_height);
//    plotMatches(prev_frame, curr_frame, matches);
    if (local_rad()<5.0f)
    {
        return false;
    }

    m_views.push_back(curr_view);
    m_fixed_mask.push_back(false);

    if (n_epi_inlr<min_matches)
    {
        std::cerr << "failed to connect current frame: Insufficient matches " << matches.size() << std::endl;
        std::exit(-1);
    }


    View::connect(*prev_view, *curr_view, matches, relPose);
    std::cout << "# matches for (" << prev_view_idx << ", " << curr_view_idx << ")  =  " << matches.size() << std::endl;

    count_connections++;
    prev_view_idx--;

    plotMatches(prev_frame, curr_frame, matches);


    // ----------------------------------------------
    // solve for next frames using global search
    // ----------------------------------------------

    int min_matches_for_skip = (win_size+1) * 50;
    while (prev_view_idx>= 0 && (curr_view_idx-prev_view_idx) <= win_size){
    //---------------------------------------------
    //   local search for prev frame
    // ---------------------------------------------
        prev_view = m_views[prev_view_idx];
        Frame &prev_frame = prev_view->frame();

        cv::Mat E;
        FeatureMatches matches;

        std::cout<<"finding init pose"<<std::endl;
        int n_epi_inlr = 0;
        Pose relPose = findInitialPose(*prev_view, *curr_view, prev_view_idx, n_epi_inlr, E, matches, min_matches_for_skip, min_set, img_width, img_height);

        if (n_epi_inlr < 100)
        {
            std::cout << "cannot connect (" << prev_view_idx << ", " << curr_view_idx << ")  -- insufficient matches: "
                      << n_epi_inlr << std::endl;
            break;
        }

        std::cout << "# matches for (" << prev_view_idx << ", " << curr_view_idx << ")  =  " << matches.size() << std::endl;


        View::connect(*prev_view, *curr_view, matches, relPose);
        count_connections++;
        prev_view_idx--;
        min_matches_for_skip = min_matches_for_skip - 50;
//        plotMatches(prev_frame, curr_frame, matches);
    }
    
    if (count_connections==0)
    {
        std::cerr << "could not connect frame!" << std::endl;
        std::exit(-1);
    }
    
    return true;
}


void ViewGraph::saveViewGraph(const std::string &filename) const
{
    cv::FileStorage file(filename, cv::FileStorage::WRITE);
    
    for (auto view: m_views)
    {
        int j = view->frame().id();
        for(const auto &x : view->connections())
        {
            auto v_ptr = x.first; // ptr to connected view to j
            auto matches_ptr = x.second; // ptr to ViewConnection object
            int i = v_ptr->frame().id();
            
            if (i < j)
            {
                const Pose &pose = matches_ptr->pose(); //get pose
                file <<"i"<< i;
                file <<"j"<< j;
                file <<"R"<< cv::Mat(pose.R());
                file <<"t"<< cv::Mat(pose.t());
            }
        }
    }
}


void ViewGraph::saveAtView(const std::string &conn_filename, const std::string &tij_conn_filename, const std::string &view_conn_filename, int target_view_idx, const int m_rad) const
{

    
    //const int m_rad = 20; // local views [ -m/rad ... -2 -1 0 1 2 ... m/rad]
    
    auto num_of_views = m_views.size();
    
    auto cam_start = MAX( 0, target_view_idx - m_rad);
    auto cam_end   = MIN( target_view_idx + m_rad, num_of_views-1);

    
    auto view_it = m_views.begin() + cam_start;
    auto view_it_end = view_it + (cam_end-cam_start);

    std::set<int> vg_views;

    std::vector<int> curr_frame_id(2);
    std::vector<int> matched_frame_idx;

    std::vector<double> posesR;
    std::vector<double> posesT;

    std::vector<double> curr_t(3);
    std::vector<int> num_matched_feature;
    std::vector<int> matched_feature_idx;
    std::vector<int> curr_matched_feature_idx(2);

    std::vector<int> tij_matched_frame_idx;
    std::vector<double> tij_posesT;

    std::vector<int> curr_num_matched_feature(1);
    std::vector<int> tij_num_matched_feature;
    std::vector<int> tij_matched_feature_idx;


    file::File f_conn = file::create(conn_filename, file::AccessFlags::TRUNCATE);


    node::Group f_conn_root_group = f_conn.root();
    node::Group connections_group = f_conn_root_group.create_group("connections_group");

    file::File tij_f_conn = file::create(tij_conn_filename, file::AccessFlags::TRUNCATE);
    node::Group tij_f_conn_root_group = tij_f_conn.root();
    node::Group tij_connections_group = tij_f_conn_root_group.create_group("tij_connections_group");

    // save connections
    for (; view_it != view_it_end; view_it++)
    {
        auto view = *view_it;
        
        int j = view->frame().id();
        for(const auto &x : view->connections())
        {
            auto v_ptr = x.first; // ptr to connected view to j
            auto matches_ptr = x.second; // ptr to ViewConnection object
            int i = v_ptr->frame().id();
            
            if (i < j)
            {
                // save tij here
                const auto &tij_pose = matches_ptr->pose(); //get pose
                const auto &tij_t = tij_pose.t();

                const auto &tij_matches = matches_ptr->matches();

                curr_frame_id[0] = i;
                curr_frame_id[1] = j;
                tij_matched_frame_idx.insert(tij_matched_frame_idx.end(), curr_frame_id.begin(), curr_frame_id.end());

                curr_num_matched_feature[0] = tij_matches.size();
                tij_num_matched_feature.insert(tij_num_matched_feature.end(), curr_num_matched_feature.begin(), curr_num_matched_feature.end());

                curr_t[0] = tij_t.val[0];
                curr_t[1] = tij_t.val[1];
                curr_t[2] = tij_t.val[2];
                tij_posesT.insert(tij_posesT.end(), curr_t.begin(), curr_t.end());

                for (auto &tij_match: tij_matches)
                {
                    curr_matched_feature_idx[0] = tij_match.queryIdx;
                    curr_matched_feature_idx[1] = tij_match.trainIdx;

                    tij_matched_feature_idx.insert(tij_matched_feature_idx.end(), curr_matched_feature_idx.begin(), curr_matched_feature_idx.end());

                }

                if (j-i==1 || j-i>30) // loop and loop-closure
                {
                    vg_views.insert(i);
                    vg_views.insert(j);

                    const auto &pose = matches_ptr->pose(); //get pose
                    const auto &matches = matches_ptr->matches();

                    std::vector<double> curr_R(9);

                    curr_frame_id[0] = i;
                    curr_frame_id[1] = j;
                    matched_frame_idx.insert(matched_frame_idx.end(), curr_frame_id.begin(), curr_frame_id.end());

                    curr_num_matched_feature[0] = matches.size();
                    num_matched_feature.insert(num_matched_feature.end(), curr_num_matched_feature.begin(), curr_num_matched_feature.end());

                    const auto &r = pose.R();
                    const auto &t = pose.t();

                    curr_R[0] = r.val[0];
                    curr_R[1] = r.val[1];
                    curr_R[2] = r.val[2];
                    curr_R[3] = r.val[3];
                    curr_R[4] = r.val[4];
                    curr_R[5] = r.val[5];
                    curr_R[6] = r.val[6];
                    curr_R[7] = r.val[7];
                    curr_R[8] = r.val[8];

                    curr_t[0] = t.val[0];
                    curr_t[1] = t.val[1];
                    curr_t[2] = t.val[2];

                    posesR.insert(posesR.end(), curr_R.begin(), curr_R.end());
                    posesT.insert(posesT.end(), curr_t.begin(), curr_t.end());


                    for (auto &match: matches)
                    {
                        curr_matched_feature_idx[0] = match.queryIdx;
                        curr_matched_feature_idx[1] = match.trainIdx;

                        matched_feature_idx.insert(matched_feature_idx.end(), curr_matched_feature_idx.begin(), curr_matched_feature_idx.end());
                    }
                }

            }
        }
    }

    //pop it to the h5 file

    node::Dataset tij_matched_frame_dset = tij_connections_group.create_dataset("tij_matched_frame_idx",
                                                                        datatype::create<std::vector<int>>(),
                                                                        dataspace::create(tij_matched_frame_idx));

    tij_matched_frame_dset.write(tij_matched_frame_idx);


    node::Dataset tij_posesT_dset = tij_connections_group.create_dataset("tij_posesT",
                                                                        datatype::create<std::vector<double>>(),
                                                                        dataspace::create(tij_posesT));

    tij_posesT_dset.write(tij_posesT);

    node::Dataset tij_matched_feature_idx_dset = tij_connections_group.create_dataset("tij_matched_feature_idx",
                                                                         datatype::create<std::vector<int>>(),
                                                                         dataspace::create(tij_matched_feature_idx));

    tij_matched_feature_idx_dset.write(tij_matched_feature_idx);

    node::Dataset tij_num_matched_feature_dset = tij_connections_group.create_dataset("tij_num_matched_feature",
                                                                              datatype::create<std::vector<int>>(),
                                                                              dataspace::create(tij_num_matched_feature));

    tij_num_matched_feature_dset.write(tij_num_matched_feature);

    node::Dataset matched_frame_dset = connections_group.create_dataset("matched_frame_idx",
                                                    datatype::create<std::vector<int>>(),
                                                    dataspace::create(matched_frame_idx));

    matched_frame_dset.write(matched_frame_idx);

    node::Dataset posesR_dset = connections_group.create_dataset("posesR",
                                                                      datatype::create<std::vector<double>>(),
                                                                      dataspace::create(posesR));

    posesR_dset.write(posesR);

    node::Dataset posesT_dset = connections_group.create_dataset("posesT",
                                                                      datatype::create<std::vector<double>>(),
                                                                      dataspace::create(posesT));

    posesT_dset.write(posesT);

    node::Dataset num_matched_feature_dset = connections_group.create_dataset("num_matched_feature",
                                                                      datatype::create<std::vector<int>>(),
                                                                      dataspace::create(num_matched_feature));

    num_matched_feature_dset.write(num_matched_feature);

    node::Dataset matched_feature_idx_dset = connections_group.create_dataset("matched_feature_idx",
                                                                            datatype::create<std::vector<int>>(),
                                                                            dataspace::create(matched_feature_idx));

    matched_feature_idx_dset.write(matched_feature_idx);

    file::File f_views = file::create(view_conn_filename, file::AccessFlags::TRUNCATE);

//
//// create a group
    node::Group f_views_root_group = f_views.root();
    node::Group f_views_group = f_views_root_group.create_group("views_group");


    std::vector<int> id_view;
    std::vector<int> curr_id_view(1);
    std::vector<int> num_kp_view;
    std::vector<int> curr_num_kp_view(1);
    std::vector<double> posesR_view;
    std::vector<double> curr_posesR_view(9);
    std::vector<double> posesT_view;
    std::vector<double> curr_posesT_view(3);

    std::vector<double> kp_view;
    std::vector<double> curr_kp_view(2);
    // save views
    for(auto view_idx: vg_views)
    {
        auto view  = m_views[view_idx];
        auto num_of_keypoints = view->frame().undistortedKeypoints().size();
        const auto &pose = view->pose();

        curr_id_view[0] = view_idx;
        id_view.insert(id_view.end(), curr_id_view.begin(), curr_id_view.end());

        curr_num_kp_view[0] = num_of_keypoints;
        num_kp_view.insert(num_kp_view.end(), curr_num_kp_view.begin(), curr_num_kp_view.end());

        const auto &r_view = pose.R();
        const auto &t_view = pose.t();

        curr_posesR_view[0] = r_view.val[0];
        curr_posesR_view[1] = r_view.val[1];
        curr_posesR_view[2] = r_view.val[2];
        curr_posesR_view[3] = r_view.val[3];
        curr_posesR_view[4] = r_view.val[4];
        curr_posesR_view[5] = r_view.val[5];
        curr_posesR_view[6] = r_view.val[6];
        curr_posesR_view[7] = r_view.val[7];
        curr_posesR_view[8] = r_view.val[8];

        posesR_view.insert(posesR_view.end(), curr_posesR_view.begin(), curr_posesR_view.end());

        curr_posesT_view[0] = t_view.val[0];
        curr_posesT_view[1] = t_view.val[1];
        curr_posesT_view[2] = t_view.val[2];

        posesT_view.insert(posesT_view.end(), curr_posesT_view.begin(), curr_posesT_view.end());


        for (auto &keypoint : view->frame().undistortedKeypoints())
        {
            curr_kp_view[0] = keypoint.pt.x;
            curr_kp_view[1] = keypoint.pt.y;
            kp_view.insert(kp_view.end(), curr_kp_view.begin(), curr_kp_view.end());

        }
    }

    //pop it to the h5 file
    node::Dataset id_view_dset = f_views_group.create_dataset("id_view",
                                                                      datatype::create<std::vector<int>>(),
                                                                      dataspace::create(id_view));

    id_view_dset.write(id_view);

    node::Dataset num_kp_view_dset = f_views_group.create_dataset("num_kp_view",
                                                               datatype::create<std::vector<int>>(),
                                                               dataspace::create(num_kp_view));

    num_kp_view_dset.write(num_kp_view);

    node::Dataset posesR_view_dset = f_views_group.create_dataset("posesR_view",
                                                               datatype::create<std::vector<double>>(),
                                                               dataspace::create(posesR_view));

    posesR_view_dset.write(posesR_view);

    node::Dataset posesT_view_dset = f_views_group.create_dataset("posesT_view",
                                                                    datatype::create<std::vector<double>>(),
                                                                    dataspace::create(posesT_view));

    posesT_view_dset.write(posesT_view);

    node::Dataset kp_view_dset = f_views_group.create_dataset("kp_view",
                                                                datatype::create<std::vector<double>>(),
                                                                dataspace::create(kp_view));

    kp_view_dset.write(kp_view);

    
//    file.close();
    
    
}


void rmat2quat(const Pose::Mat3 &R, Pose::Vec4 &Q)
{
    //double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
    double trace = R(0,0) + R(1,1) + R(2,2);
    
    if (trace > 0.0)
    {
        double s = sqrt(trace + 1.0);
        Q(3) = s * 0.5;
        s = 0.5 / s;
        Q(0) = (R(2,1) - R(1,2)) * s;
        Q(1) = (R(0,2) - R(2,0)) * s;
        Q(2) = (R(1,0) - R(0,1)) * s;
    }
    else
    {
        int i = R(0,0) < R(1,1) ? ( R(1,1) < R(2,2) ? 2 : 1) : (R(0,0) < R(2,2) ? 2 : 0);
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;
        
        double s = sqrt(R(i, i) - R(j,j) - R(k,k) + 1.0);
        Q(i) = s * 0.5;
        s = 0.5 / s;
        
        Q(3) = (R(k,j) - R(j,k)) * s;
        Q(j) = (R(j,i) + R(i,j)) * s;
        Q(k) = (R(k,i) + R(i,k)) * s;
    }
}


void ViewGraph::savePoses(const std::string &filename) const
{
    Pose::Vec4 q;


    file::File f = file::create(filename, file::AccessFlags::TRUNCATE);

    node::Group root_group = f.root();
    node::Group poses_group = root_group.create_group("poses_group");

    std::vector<double> q_vec;
    std::vector<double> curr_q_vec(4);
    std::vector<double> t_vec;
    std::vector<double> curr_t_vec(3);
    std::vector<int> curr_id(1);
    std::vector<int> idx;


    for (auto view: m_views)
    {
        const auto &pose = view->pose();
        const auto &id = view->frame().id();
        const auto &t = pose.t();
        rmat2quat(pose.R(), q);

        curr_q_vec[0] = q(3);
        curr_q_vec[1] = q(0);
        curr_q_vec[2] = q(1);
        curr_q_vec[3] = q(2);

        q_vec.insert(q_vec.end(), curr_q_vec.begin(), curr_q_vec.end());

        curr_t_vec[0] = t(0);
        curr_t_vec[1] = t(1);
        curr_t_vec[2] = t(2);

        t_vec.insert(t_vec.end(), curr_t_vec.begin(), curr_t_vec.end());

        curr_id[0] = id;
        idx.insert(idx.end(), curr_id.begin(), curr_id.end());

    }

    //pop into f5
    node::Dataset q_dset = poses_group.create_dataset("q",
                                                            datatype::create<std::vector<double>>(),
                                                            dataspace::create(q_vec));

    q_dset.write(q_vec);

    node::Dataset t_dset = poses_group.create_dataset("t",
                                                       datatype::create<std::vector<double>>(),
                                                       dataspace::create(t_vec));

    t_dset.write(t_vec);

    node::Dataset idx_dset = poses_group.create_dataset("idx",
                                                       datatype::create<std::vector<int>>(),
                                                       dataspace::create(idx));

    idx_dset.write(idx);

}


void ViewGraph::fixPose(int idx, Pose &new_pose)
{
    assert(m_fixed_mask.size() == m_views.size());
    
    m_fixed_mask[idx] = true;
    
    auto &view = m_views[idx];
    Pose &pose = view->pose();
    pose = new_pose;
    
    assert(m_fixed_mask.size() == m_views.size());
}

bool ViewGraph::isPoseFixed(int idx) const
{
    return m_fixed_mask[idx];
}

int ViewGraph::countFixedPoses() const
{
    int resp = 0;
    for (const auto x: m_fixed_mask)
    {
        if (x) resp++;
    }
    return resp;
}


void ViewGraph::rotAvg(int winSize)
{
    assert(winSize > 2);
    const long &m = m_views.size();
    
    // upgrade winSize if few views
    winSize = MIN( (int)m, winSize);
    if (winSize<2)
    {
        return; // no variables to optimise
    }
    
    // ---------------------------------------
    // Retrieve local connections
    // ---------------------------------------
    irotavg::I_t I;

    std::vector< Pose::Vec4 > qq_vec;
    
    std::set<int> vertices;

    for (long t = m-winSize; t<m; t++)
    {
        const auto &view = m_views[t];
        
        Pose::Vec4 q;
    
        int j = view->frame().id();
        for(const auto &x : view->connections())
        {
            auto v_ptr = x.first;
            auto matches_ptr = x.second;
            int i = v_ptr->frame().id();
            
            if (i < j)
            {
                const Pose &pose = matches_ptr->pose(); //get pose
                
                I.push_back( std::make_pair(i,j) ); // indices must be reacomodated!
                vertices.insert(i);
                vertices.insert(j);
                
                rmat2quat(pose.R(), q);
                qq_vec.push_back(q);
            }
        }
    }
    
    const long &num_of_edges = qq_vec.size();
    const long &num_of_vertices = vertices.size();
    //int f = ((int)num_of_vertices) - winSize;

    if (num_of_edges < winSize)
    {
        return;  // skip optimising if insufficient edges
    }
    
    if (num_of_vertices < winSize)
    {
        return;  // skip optimising if graph is unconnected
    }

    // acomodate indices in I
    std::map<int,int> vertex_to_idx, idx_to_vertex;
    
    // count the fixed cameras
    
    // we fix cameras out of the window
    int f = ((int)num_of_vertices) - winSize;
    
    assert(f>=0);
    
    // count non-fixed cameras in the window.
    for (auto const& x : vertices)
    {
        if ( x >= m-winSize && m_fixed_mask[x] )
            f++;
    }

    int t = 0; // new idx for fixed rotations
    int k = f; // new idx for non-fixed rotations

    for (auto const& x : vertices)
    {
        if (x >= m-winSize) // cam in the window
        {
            if ( isPoseFixed(x) )
            {
                idx_to_vertex[t] = x;
                vertex_to_idx[x] = t++;
            }
            else
            {
                idx_to_vertex[k] = x;
                vertex_to_idx[x] = k++;
            }
        }
        else // always fix poses out of the window
        {
            idx_to_vertex[t] = x;
            vertex_to_idx[x] = t++;
        }
    }
    
    for (auto &c: I)
    {
        c.first  = vertex_to_idx[c.first];
        c.second = vertex_to_idx[c.second];
    }
    
    // make Q
    irotavg::Mat Q(num_of_vertices, 4);
    Pose::Vec4 q;
    for (const auto &x : vertices)
    {
        const auto &view = m_views[x];
        Pose &pose = view->pose();
        rmat2quat(pose.R(), q);
        Q.row(vertex_to_idx[x]) << q(0), q(1), q(2), q(3);
    }
    
    if (f == 0)
    {
        Q.row(0) << 0, 0, 0, 1;
        f = 1;
    }
    
    // make QQ
    irotavg::Mat QQ(num_of_edges, 4);
    for (long i=0; i<num_of_edges; i++)
    {
        const auto &q = qq_vec[i];
        QQ.row(i) << q(0), q(1), q(2), q(3);
    }
    
    // comment next line for no initialisation -- just refine
    // irotavg::init_mst(Q, QQ, I, f);

    // make A
    irotavg::SpMat A = irotavg::make_A((int)num_of_vertices, f, I);
    
    const double change_th = .001;
    
    const int l1_iters = 100;
    int l1_iters_out;
    double l1_runtime;
    irotavg::l1ra(QQ, I, A, Q, f, l1_iters, change_th, l1_iters_out, l1_runtime);

    const int irls_iters = 100;
    int irls_iters_out;
    double irls_runtime;
    irotavg::Vec weights(num_of_edges);
    irotavg::Cost cost = irotavg::Cost::Geman_McClure;
    double sigma = 5*M_PI/180.0;
    
    irotavg::irls(QQ, I, A, cost, sigma, Q, f, irls_iters, change_th,
         weights, irls_iters_out, irls_runtime);
    
    // upgrade poses for the window
    for (k=f; k<num_of_vertices; k++)
    {
        auto &view = m_views[ idx_to_vertex[k] ];

        Pose &pose = view->pose();
        
        irotavg::Quat q(Q(k,3), Q(k,0), Q(k,1), Q(k,2));
        q = q.normalized();
        
        irotavg::Mat R = q.toRotationMatrix();
        R.transposeInPlace(); // opencv is row-major
        Pose::Mat3 R_cv(R.data());
        
        pose.setR(R_cv);
    }
}


bool View::connect(View &v1, View &v2, FeatureMatches matches, Pose rel_pose)
{
    assert(matches.size()>4); //
    if (v1.m_connections.count(&v2)>0)
    {
        assert(v2.m_connections.count(&v1)>0); // we must have an undirected graph
        return false;
    }
    
    // Create a ViewConnection object
    ViewConnection *connection = new ViewConnection(v1, v2,
                                                    std::move(matches),
                                                    std::move(rel_pose));

    v1.m_connections[&v2] = connection;
    v2.m_connections[&v1] = connection;
    
    return true;
}

