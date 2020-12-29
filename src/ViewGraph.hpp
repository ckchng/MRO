

#ifndef ViewGraph_hpp
#define ViewGraph_hpp

#include <stdio.h>
#include <vector>

#include "Frame.hpp"
#include "Pose.hpp"
#include "View.hpp"

#include "l1_irls.hpp"

namespace irotavg
{
class ViewGraph
{
    
public:
    
    ViewGraph(std::vector<float> scale_sigma_squares):
    m_scale_sigma_squares(scale_sigma_squares)
    {}
    
    // Add a frame to the view graph
    // return true if frame is added, otherwie false
    // win_size     --  window size of recent to try to connect with
    // min_matches  --  min number of matches for adding a connection
    bool processFrame(Frame &frame,
                      const int min_set,
                      const int img_width,
                      const int img_height,
                      const int win_size=10,
                      const int min_matches=100);

    
    View &currentView()
    {
        return *m_views.back();
    }
    
    void saveViewGraph(const std::string &filename) const;
    
    void saveAtView(const std::string &filename, const std::string &tij_filename, const std::string &view_filename, const int viewIdx, const int rad) const;
    
    void savePoses(const std::string &filename) const;
    
    // refine rotations by using rotation averaging
    void rotAvg(const int winSize);
    
    void fixPose(int idx, Pose &pose);
    
    bool isPoseFixed(int idx) const;
    
    int countFixedPoses() const;
    
    
    Pose findInitialPose(View &v1, View &v2, const int prev_view_idx, int &n_epi_inlrs,
                         cv::Mat &E, FeatureMatches &matches, int min_matches, const int min_set,
                          const int img_width, const int img_height);
    
    
    //  iteratively refine pose by alternating between
    //  finding matches with E
    //  and solving F from matches
    int refinePose (Frame &f1, Frame &f2, const int min_set, Pose &pose, cv::Mat &E, FeatureMatches &matches) const;

    // -----------------------------------------------------------------------------------
    // ---- loop closure functions
    // -----------------------------------------------------------------------------------
    
    bool detectLoopCandidates(View &view, std::vector<View*> &candidates);
    
    typedef std::pair<std::set<View*>, int> ConsistentGroup;
    
    bool checkLoopConsistency(const std::vector<View*> &loop_candidates,
                              std::vector<View*> &consistent_candidates,
                              std::vector<ConsistentGroup> &consistent_groups,
                              std::vector<ConsistentGroup> &prev_consistent_groups,
                              const int covisibility_consistency_th=3); //3
    
    int findORBMatchesByBoW(Frame &f1, Frame &f2,
                            std::vector<std::pair<int,int> > &matches,
                            const double nnratio) const;

    Pose findRelativePose(Frame &f1, Frame &f2,
                          FeatureMatches &matches,
                          int &n_epi_inlrs, cv::Mat &mask,
                          cv::Mat &E, double th=1.0) const;


    Pose findRelativePose_with_eigsolver(Frame &f1, Frame &f2,
                          const std::vector<double> prevRrel, int &ransac_iter,
                          const int min_set, const int max_rsc_iter,
                          FeatureMatches &matches,
                          std::vector<int> &inlier_vec,
                          int &n_epi_inlrs, cv::Mat &mask,
                          cv::Mat &E, double th=1.0) const;
    
    void filterMatches(FeatureMatches &matches, const cv::Mat &inlrs_mask, int n_epi_inlrs) const;
    void filterMatches_w_inlr_id(FeatureMatches &matches, const std::vector<int> inlier_vec, int n_epi_inlrs) const;
    
    double local_rad() const { return m_local_rad; }
    
private:
    
    bool checkDistEpipolarLine(const cv::KeyPoint &kp1,
                               const cv::KeyPoint &kp2,
                               const cv::Mat &F12) const;
    
    int findORBMatches(Frame &f1, Frame &f2, cv::Mat F12, 
                       std::vector<std::pair<int,int> > &matches) const;
    
    //TODO: set input arguments as const
    bool findPose(View &v1, View &v2, View &pivot_view, const int min_set,
                  const std::vector<int> pivot2current_map, Pose &pose,
                  cv::Mat &E, FeatureMatches &matches);
    
    std::vector<View*> m_views;
    std::vector<float> m_scale_sigma_squares;
    
    std::vector<bool> m_fixed_mask; //mask for fixed views
    
    //double m_local_rad = 85;
    double m_local_rad = 45;
    
};

}
#endif /* ViewGraph_hpp */
