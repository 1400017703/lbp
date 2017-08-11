#include <vector>
#include <utility>
#include <cv.h>
#include <highgui.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include "FaceDetector.h"
#include "LBPFeatureExtractor.h"
#include "FaceNormalizer.h"
#include "sirius_util.h"

using namespace std;
using namespace cv;

const std::vector<int> scales = {300, 212, 150, 106, 75};
const int minSize = 100;
const int scale = 250;
const int patchSize = 10;
const int gridNumX = 4;
const int gridNumY = 4;
const int lbpDim = 59;
const int landmarkNum = 25;
const double eyeDistanceX = 2;
const double eyeDistanceUp = 1.5;
const double eyeDistanceDown = 2.5;
//const int totalDim = scales.size()*gridNumX*gridNumY*lbpDim*landmarkNum;
//TODO
const int totalDim = 118000;

Mat* FaceDetection(Mat &image, FaceDetector &faceDetector);

int* ExtractLBP(Mat *pImage, LBPFeatureExtractor &lbpFeatureExtractor, dlib::shape_predictor &pose_model);
