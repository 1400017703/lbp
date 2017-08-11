#include "PreProcess.h"

Mat* FaceDetection(Mat &image, FaceDetector &faceDetector){
    vector<Rect> outputFaces;
    faceDetector.detect(image, outputFaces, Size(minSize, minSize));
    if (outputFaces.size()==0) return NULL;
    Rect largestFace(0,0,0,0);
    for (vector<Rect>::const_iterator r = outputFaces.begin(); r != outputFaces.end(); r++)
        if (r->width > largestFace.width) largestFace = *r;
    copyMakeBorder(image, image, int(largestFace.height), int(largestFace.height),
        int(largestFace.width), int(largestFace.width), BORDER_CONSTANT, Scalar(0));
    Mat *pImage = new Mat;
    *pImage = image(Rect(largestFace.x + largestFace.width/2, largestFace.y + largestFace.height/2,
        largestFace.width*2, largestFace.height*2));
    resize(*pImage, *pImage, Size(scale, scale));
    return pImage;
}

int* ExtractLBP(Mat *pImage, LBPFeatureExtractor &lbpFeatureExtractor, dlib::shape_predictor &pose_model){
    vector< pair<double, double> > points;
    dlib::cv_image<dlib::bgr_pixel> cimg(*pImage);
    dlib::full_object_detection shape = pose_model(cimg, dlib::rectangle(pImage->rows/4,pImage->cols/4,pImage->rows*3/4, pImage->cols*3/4));
    pair<double,double> point;
    point.first = 0.5*(shape.part(36).x()+shape.part(39).x());
    point.second = 0.5*(shape.part(36).y()+shape.part(39).y());
    points.push_back(point);
    point.first = 0.5*(shape.part(42).x()+shape.part(45).x());
    point.second = 0.5*(shape.part(42).y()+shape.part(45).y());
    points.push_back(point);
    const int landmarkPosition[] = {17,21,22,26,36,39,42,45,29,31,35,57,60,64,19,24,27,33,50,52,28,30,68};
    for(int i = 0; i<landmarkNum - 2; i++){
        point.first = shape.part(landmarkPosition[i]).x();
        point.second = shape.part(landmarkPosition[i]).y();
        points.push_back(point);
    }
    Mat faceImage;
    vector<pair<double,double> > newPoints;
    FaceNormalizer faceNormalizer(eyeDistanceX, eyeDistanceUp, eyeDistanceDown);
    if (faceNormalizer.normalize(*pImage, points, faceImage, newPoints)) {
        delete pImage;
        int *feature = new int[totalDim];
        lbpFeatureExtractor.extractAt(faceImage, newPoints, feature);
        return feature;
    }
    else{
        delete pImage;
        return NULL;
    }
}

