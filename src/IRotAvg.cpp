#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
#include <h5cpp/hdf5.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>


#include "Frame.hpp"
#include "ORBExtractor.hpp"
#include "ViewGraph.hpp"
#include "SequenceLoader.hpp"
#include "ViewDatabase.hpp"


using namespace irotavg;
using namespace hdf5;
using namespace opengv;

CameraParameters cam_pars;
ORB_SLAM2::ORBextractor *orb_extractor;

void config(const std::string &filename)
{
    // check settings file
    cv::FileStorage settings(filename.c_str(), cv::FileStorage::READ);
    if(!settings.isOpened())
    {
        std::cerr << "Failed to open settings file: " << filename << std::endl;
        std::exit(-1);
    }

    //--------------------------------------------------------------------------------------------
    // Camera Parameters
    //--------------------------------------------------------------------------------------------
    CameraParameters::Intrinsic_type K = CameraParameters::Intrinsic_type::eye();

    const double fx = settings["Camera.fx"];
    const double fy = settings["Camera.fy"];
    const double cx = settings["Camera.cx"];
    const double cy = settings["Camera.cy"];

    K(0,0) = fx;
    K(1,1) = fy;
    K(0,2) = cx;
    K(1,2) = cy;

    CameraParameters::Dist_type dist_coef;

    dist_coef(0) = settings["Camera.k1"];
    dist_coef(1) = settings["Camera.k2"];
    dist_coef(2) = settings["Camera.p1"];
    dist_coef(3) = settings["Camera.p2"];


    cam_pars = CameraParameters(K,dist_coef);

    //--------------------------------------------------------------------------------------------
    // ORB Parameters
    //--------------------------------------------------------------------------------------------
    int n_features = settings["ORBextractor.nFeatures"];
    float scale_factor = settings["ORBextractor.scaleFactor"];
    int n_levels = settings["ORBextractor.nLevels"];
    int ini_th_FAST = settings["ORBextractor.iniThFAST"];
    int min_th_FAST = settings["ORBextractor.minThFAST"];

    orb_extractor = new ORB_SLAM2::ORBextractor(n_features,scale_factor,
                                                n_levels,ini_th_FAST,min_th_FAST);
}

// move to some util class...
void myPlotMatches(Frame &prev_frame, Frame &curr_frame, FeatureMatches &matches)
{
    auto im1 = prev_frame.getImage();
    auto im2 = curr_frame.getImage();
    auto kps1 = prev_frame.keypoints();
    auto kps2 = curr_frame.keypoints();

    cv::Mat im_matches;
    cv::drawMatches(im1, kps1, im2, kps2, matches, im_matches);
    double s = 1;
    cv::Size size(s*im_matches.cols,s*im_matches.rows);
    resize(im_matches,im_matches,size);
    cv::imshow("matches after ransac", im_matches);
    cv::waitKey(1);
}



void saveSelectedFramesIds(const std::string &filename, std::vector<int> &selected)
{
    Pose::Vec4 q;

//    std::ofstream fs(filename);

    file::File f = file::create(filename,file::AccessFlags::TRUNCATE);
    node::Group root_group = f.root();
    node::Group sel_id_group = root_group.create_group("sel_id_group");

    std::vector<int> sel_id;
    std::vector<int> curr_sel_id(1);

    for (auto id: selected)
    {
        curr_sel_id[0] = id;
        sel_id.insert(sel_id.end(), curr_sel_id.begin(), curr_sel_id.end());

    }

    node::Dataset sel_id_dset = sel_id_group.create_dataset("sel_id",
                                                            datatype::create<std::vector<int>>(),
                                                            dataspace::create(sel_id));

    sel_id_dset.write(sel_id);
}



int main(int argc, const char *argv[])
{
    std::string licence_notice =
            "MRO Copyright (C) 2020 CK Chng\n"
            "MRO comes with ABSOLUTELY NO WARRANTY.\n"
            "    This is free software, and you are welcome to redistribute it\n"
            "    under certain conditions; visit\n"
            "    https://github.com/ckchng/MRO#License for details.\n"
            "\n";


    const cv::String keys =
            "{help h usage ?   |      | print this message            }"
            "{@orb_vocabulary  |<none>| orb vocabulary                }"
            "{@config          |<none>| config file                   }"
            "{@sequence_path   |<none>| path to images                }"
            "{@sel_id  |<none>| orb vocabulary                }"
            "{@conn          |<none>| config file                   }"
            "{@tij_conn   |<none>| path to images                }"
            "{@view          |<none>| config file                   }"
            "{@poses   |<none>| path to images                }"
            "{@MIN_SET  |<none> | MINSET               }"
            "{@vg_win_size  |<none> | vg_win_size               }"
            "{@img_width  |<none> | img_width               }"
            "{@img_height  |<none> | img_height               }"
            "{image_ext        |.png | image extension               }"
            "{timestamp_offset |0     | image's name timestamp offset }"
            "{gt               |      | ground truth                  }"

    ;

    std::cout << licence_notice << std::endl;

    std::cout << "OpenCV version: "
              << CV_MAJOR_VERSION << "."
              << CV_MINOR_VERSION << "."
              << CV_SUBMINOR_VERSION
              << std::endl;

    std::cout << "Eigen version: "
              << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << "."
              << CV_SUBMINOR_VERSION
              << std::endl;

    const bool detect_loop_closure = true;

    const int rotavg_win_size = 10;
    const int vg_min_matches = 200;

    cv::CommandLineParser parser(argc, argv, keys);

    const std::string vocab_filename ( parser.get<cv::String>(0) );
    const std::string config_filename( parser.get<cv::String>(1) );
    const std::string sequence_path  ( parser.get<cv::String>(2) );
    const std::string sel_id_path  ( parser.get<cv::String>(3) );
    const std::string conn_path  ( parser.get<cv::String>(4) );
    const std::string tij_conn_path  ( parser.get<cv::String>(5) );
    const std::string view_path  ( parser.get<cv::String>(6) );
    const std::string poses_path  ( parser.get<cv::String>(7) );
    const int MIN_SET  ( parser.get<int>(8) );
    const int vg_win_size ( parser.get<int>(9) );
    const int img_width ( parser.get<int>(10) );
    const int img_height ( parser.get<int>(11) );

    const std::string image_ext( parser.get<cv::String>("image_ext") );
    const int timestamp_offset = parser.get<int>("timestamp_offset");

    //--------------------------------------------------------------------------------------------
    // ORB Vocabulary
    //--------------------------------------------------------------------------------------------

    auto &orb_vocab = ORBVocabulary::instance();
    orb_vocab.load(vocab_filename);

    bool gt_provided = parser.has("gt");
    std::string gt_file;
    if (gt_provided)
    {
        gt_file = parser.get<cv::String>("gt");
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    // ----------------------------------------------
    // read GT -- when fixing some rotations...
    // ----------------------------------------------
    std::vector<Pose::Mat3> gt_rots;
    if (gt_provided)
    {
        std::ifstream myfile (gt_file);
        if (!myfile.is_open())
        {
            std::cerr << "Unable to open file " << gt_file << std::endl;
            std::exit(-1);
        }
        double r1, r2, r3, r4, r5, r6, r7, r8, r9;
        while (myfile >> r1 >> r2 >> r3 >> r4 >> r5 >> r6 >> r7 >> r8 >> r9)
        {
            //gt is given as orientations so I need to transpose
            Pose::Mat3 R(r1, r2, r3,
                         r4, r5, r6,
                         r7, r8, r9); // opencv is row-major!

            gt_rots.push_back(R);
        }
        myfile.close();
    }


    config(config_filename);

    std::cout<< "K:\n";
    std::cout<< cam_pars.intrinsic() << std::endl;

    std::cout<< "dist coefs:\n";
    std::cout<< cam_pars.dist_coef() << std::endl;

    SequenceLoader loader(sequence_path, image_ext, timestamp_offset);

    ViewGraph view_graph(orb_extractor->GetScaleSigmaSquares());

    // to check consistency for loop clodure
    std::vector<ViewGraph::ConsistentGroup> prev_consistent_groups;
    auto &db = ViewDatabase::instance();

    std::cout<< "initialising database..."<<std::endl;
    db.init();


    bool is_camera_init = false; //flag for camera initialization

    std::vector<int> selected_frames;

    clock_t t_prev = clock();


    int id = 0;
    int count = 0;
    const int sampling_step = 1; //5


    for (auto frame : loader)
    {
        if (count++%sampling_step != 0) // sampling
        {
            continue;
        }

        clock_t tic = clock(), toc;

        std::string impath = frame.second.string();

        // Set camera parameters
        if (!is_camera_init)
        {
            cv::Mat im = cv::imread(impath, cv::IMREAD_GRAYSCALE);   // Read the file
            Camera::instance().init(cam_pars, im);
            is_camera_init = true;
        }

        // Create a Frame object
        Frame f(id, impath, *orb_extractor);

        toc = clock();
        double frame_creation_time = (double)(toc-tic)/CLOCKS_PER_SEC;

        tic = clock();

        bool is_frame_selected = view_graph.processFrame(f, MIN_SET, img_width, img_height, vg_win_size, vg_min_matches);

        if (!is_frame_selected)
        {
            std::cout<<"skipping frame - local rad = "<<view_graph.local_rad()<<std::endl<<std::endl;
            continue;
        }

        selected_frames.push_back(count);


        View &view = view_graph.currentView();

        // -------- loop closure
        bool loop_new_connections = false;

        if (detect_loop_closure){

            std::vector<View*> loop_candidates;
            if ( id%1==0 && view_graph.detectLoopCandidates(view, loop_candidates) ) {
                std::vector<View*> consistent_candidates;
                std::vector<ViewGraph::ConsistentGroup> consistent_groups;

                if (view_graph.checkLoopConsistency(loop_candidates, consistent_candidates,
                                                    consistent_groups, prev_consistent_groups)) {
                    std::cout << " * * * loop closure detected * * *\n" << std::endl;

                    for (View *prev_view : consistent_candidates) {
                        const int min_matches = 200;

                        //find matches
                        double nnratio = .9;
                        std::vector<std::pair<int, int> > matched_pairs;
                        view_graph.findORBMatchesByBoW(prev_view->frame(), f,
                                                       matched_pairs, nnratio);
                        FeatureMatches matches;
                        for (auto &match_pair: matched_pairs)
                            matches.push_back(cv::DMatch(match_pair.first, match_pair.second, 0));

                        //find pose
                        int inlrs;
                        cv::Mat inlrs_mask;
                        inlrs_mask = cv::Mat();
                        cv::Mat E;
                        std::vector<int> inlier_vec;
                        std::cout << "finding relative pose" << std::endl;

                        std::vector<double> prevRrel_vec(9);
                        prevRrel_vec[0] = 1;
                        prevRrel_vec[1] = 0;
                        prevRrel_vec[2] = 0;

                        prevRrel_vec[3] = 0;
                        prevRrel_vec[4] = 1;
                        prevRrel_vec[5] = 0;

                        prevRrel_vec[6] = 0;
                        prevRrel_vec[7] = 0;
                        prevRrel_vec[8] = 1;

                        // use the result of current frame
                        if (f.id() > 1) {
                            for (const auto &x : view.connections()) {
                                auto v_ptr = x.first; // ptr to connected view to j
                                auto matches_ptr = x.second; // ptr to ViewConnection object
                                int i = v_ptr->frame().id();

                                if (i == f.id() - 1) {
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

                        int ransac_iter = 0;
                        Pose relPose = view_graph.findRelativePose_with_eigsolver(prev_view->frame(), f, prevRrel_vec,
                                                                                  ransac_iter, MIN_SET, 10000,
                                                                                  matches, inlier_vec, inlrs,
                                                                                  inlrs_mask, E);
                        view_graph.filterMatches_w_inlr_id(matches, inlier_vec, inlrs);
                        inlier_vec.clear();

                        if (ransac_iter > 5000) {
                            continue;
                        }


                        if (matches.size() < min_matches){
                            std::cout << " can not connect, # matches : " << matches.size() << std::endl;
                            continue;

                        }



                        View::connect(*prev_view, view, matches, relPose);
                        std::cout << "   new connection: ( " << prev_view->frame().id() << ", " << view.frame().id()
                                  << " ) ";
                        std::cout << " # matches: " << matches.size() << std::endl;
                        loop_new_connections = true;

                    }

                }
                prev_consistent_groups = std::move(consistent_groups);
            }
            db.add(&view);
        }
        // -------- end loop closure

        toc = clock();
        double frame_processing_time = (double)(toc-tic)/CLOCKS_PER_SEC;

        tic = clock();
        bool add_correction = gt_provided && id%20==0;
        if (add_correction)
        {
            Pose pose_gt;
            pose_gt.setR( gt_rots[id*sampling_step] );

            view_graph.fixPose(id, pose_gt);
            std::cout<<"Fixing pose for view id " << id << std::endl;
        }


        if (loop_new_connections || add_correction)
        {
            view_graph.rotAvg(5000000); //global
        }
        else
        {
            view_graph.rotAvg(rotavg_win_size); //local
        }
        toc = clock();
        double rotavg_time = (double)(toc-tic)/CLOCKS_PER_SEC;

        printf("frame %d  -- runtimes: frame creation %.3fl; frame processing %.3f, rotavg %.3f\n",
               id, frame_creation_time, frame_processing_time, rotavg_time);

        // save view graph every 10 seconds
        clock_t t_current = clock();
        double dt = double(t_current-t_prev) / CLOCKS_PER_SEC; // in seconds

        if (dt > 10.0)
        {
            t_prev = t_current;
            view_graph.savePoses(poses_path);

            view_graph.saveAtView(conn_path, tij_conn_path, view_path, -1, 1000000); //-1 to save it at the current view!
            saveSelectedFramesIds(sel_id_path, selected_frames);
        }


        id++;
    }

    view_graph.savePoses(poses_path);

    view_graph.saveAtView(conn_path, tij_conn_path, view_path, -1, 1000000); //-1 to save it at the current view!
    saveSelectedFramesIds(sel_id_path, selected_frames);

    return 0;
}
