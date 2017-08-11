OPENCV_HOME = /usr/include/opencv/
DLIB_HOME = ../include/dlib-19.4/

CC = g++
CFLAG = -std=c++11
LIB = -L/usr/local/lib -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio
INC = -I../include/ -I$(OPENCV_HOME) -I$(DLIB_HOME)
OBJ = ArgumentParser.o FaceDetector.o LBPFeatureExtractor.o FaceNormalizer.o sirius_util.o PreProcess.o
EXE = test demo pca

all:test demo pca

test: $(DLIB_HOME)dlib/all/source.cpp test.cpp $(OBJ)
	$(CC) -o $@ $< -lpthread -lX11  $@.cpp $(OBJ) $(LIB) $(INC) $(CFLAG)
	cp $@ ../bin/

demo: $(DLIB_HOME)dlib/all/source.cpp demo.cpp $(OBJ)
	$(CC) -o $@ $< -lpthread -lX11  $@.cpp $(OBJ) $(LIB) $(INC) $(CFLAG)
	cp $@ ../bin/

pca: $(DLIB_HOME)dlib/all/source.cpp pca.cpp $(OBJ)
	$(CC) -o $@ $< -lpthread -lX11  $@.cpp $(OBJ) $(LIB) $(INC) $(CFLAG)
	cp $@ ../bin/


%.o: %.cpp ../include/%.h
	$(CC) -c $< $(LIB) $(INC) $(CFLAG)

clean:
	rm -rf *.o $(EXE)
