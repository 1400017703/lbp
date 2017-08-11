#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "PreProcess.h"

const int totalSample = 10000;
const int newDim = 5000;
const string DataPath = "../data/pcaimgs/";

int main(){
    int cnt = 0;
    FaceDetector faceDetector("../data/fdetector_model.dat");
    LBPFeatureExtractor lbpFeatureExtractor(scales, patchSize, gridNumX, gridNumY, true);
    dlib::shape_predictor pose_model;
    dlib::deserialize("../data/shape_predictor_68_face_landmarks.dat") >> pose_model;
    Mat pcaSet(totalSample, totalDim, CV_32FC1);
    while(1){
        stringstream stream;
        stream << cnt << ".jpg";
        string str;
        stream >> str;
        Mat image = imread((DataPath+str).c_str());
        Mat * pImage = FaceDetection(image, faceDetector);
	    if (pImage){
            int  *testFeature = ExtractLBP(pImage, lbpFeatureExtractor, pose_model);
	        if (testFeature){
	            for (int i=0; i<totalDim; i++)
                    pcaSet.at<float>(cnt,i) = testFeature[i]/100.0;
		        delete testFeature;
		        cnt++;
		        if (cnt==totalSample) break;
	        }
	    }
    }
    PCA pca(pcaSet, Mat(), CV_PCA_DATA_AS_ROW);
    Mat mean = pca.mean;
    Mat eigenVectors = pca.eigenvectors.t();
    ofstream fpca("../data/pca.dat");
    for (int i=0; i<totalDim; i++)
        fpca << mean.at<float>(0,i) << " ";
    fpca << endl;
    for (int i=0; i<totalDim; i++){
        for (int j=0; j<newDim; j++)
            fpca << eigenVectors.at<float>(i,j) << " ";
        cout << endl;
    }
    fpca.close();
    return 0;
}
