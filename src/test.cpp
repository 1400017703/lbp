#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "PreProcess.h"

const int Persons = 8;
const int Samples = 4;
const int totalFeature = Persons * Samples;
const string DataPath = "../data/";
//int feature[totalFeature][totalDim];
const int newDim = 32;

int main(int argc, const char **argv){
    Mat pcaSet(totalFeature, totalDim, CV_32FC1);
    for (int i=1; i<=Persons; i++)
        for (int j=1; j<=Samples; j++){
            stringstream stream;
            stream << i << '/' << j << ".lbp";
            string str;
            stream >> str;
            ifstream flbp((DataPath + str).c_str());
            int tmp = (i-1)*Samples + j - 1;
            for (int k=0; k<totalDim; k++)
                flbp >> pcaSet.at<float>(tmp,k);
                //flbp >> feature[tmp][k];
            flbp.close();
        }
    PCA pca(pcaSet, Mat(), CV_PCA_DATA_AS_ROW);
    Mat mean = pca.mean;
    Mat eigenVectors = pca.eigenvectors.t();
    Mat dst = pca.project(pcaSet);
    FaceDetector faceDetector("../data/fdetector_model.dat");
    LBPFeatureExtractor lbpFeatureExtractor(scales, patchSize, gridNumX, gridNumY, true);
    dlib::shape_predictor pose_model;
    dlib::deserialize("../data/shape_predictor_68_face_landmarks.dat") >> pose_model;
    clock_t time1 = clock();
    Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat * pImage = FaceDetection(image, faceDetector);
    clock_t time2 = clock();
    double sec = double(time2 - time1)/CLOCKS_PER_SEC;
    cout << "detection:" << sec << endl;
    if (pImage){
        int  *testFeature = ExtractLBP(pImage, lbpFeatureExtractor, pose_model);
        clock_t time3 = clock();
        sec = double(time3 - time2)/CLOCKS_PER_SEC;
        cout << "lbp:" << sec << endl;
        if (testFeature){
            Mat feature = Mat(1, totalDim, CV_32FC1);
            for (int i=0; i< totalDim; i++)
                feature.at<float>(0,i) = testFeature[i];
            feature = feature - mean;
            Mat pcaFeature = feature * eigenVectors;
            for (int i=0; i<totalFeature; i++){
                float tmp = 0;
                /*for (int j=0; j<totalDim; j++){
                    float t = testFeature[j] - feature[i][j];
                    if (t!=0){
                        t*=t;
                        tmp += t/(testFeature[j] + feature[i][j]);
                    }
                }*/
                for (int j=0; j<newDim; j++){
                    float t = pcaFeature.at<float>(0,j) - dst.at<float>(i,j);
                    tmp += t*t;
                }
                cout << tmp << endl;
            }
            delete testFeature;
        }
        sec = double(clock() - time3)/CLOCKS_PER_SEC;
        cout << "classification:" << sec << endl;
    }
    return 0;
}
