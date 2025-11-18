# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: fractal.cu fractal.h
	$(CXX) fractal.cu --std c++17 `pkg-config opencv --cflags --libs` -o fractal.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcublas

run:
	./fractal.exe > output.txt

clean:
	rm -f fractal.exe rm -f out_data/*