#include <iostream>
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
    VideoCapture cap(0);
    while(true){
        Mat frame;
        cap >> frame;
        imshow("",frame);
        Mat *pImage = FaceDetection(frame, faceDetector);
        if(pImage){
            int *testFeature = ExtractLBP(pImage, lbpFeatureExtractor, pose_model);
            if (testFeature){
                Mat feature = Mat(1, totalDim, CV_32FC1);
                for (int i=0; i<totalDim; i++)
                    feature.at<float>(0,i) = testFeature[i];
                feature = feature - mean;
                Mat pcaFeature = feature * eigenVectors;
                int id = -1;
                float min = 999999999;
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
                    if (tmp < min){
                        id = i;
                        min = tmp;
                    }
                }
                delete testFeature;
                if (min<70000){
                    cout << "person:" << (id/4+1) << endl;
                    cout << "distance:" << min << endl;
                }
                else cout << "unknown:" << (id/4+1) << endl << "distance:" << min << endl;
            }
        }
        waitKey(3);
    }
    return 0;
}
