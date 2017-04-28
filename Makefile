all:
	nvcc -arch=sm_20 pa5.cu ppmFile.c -o pa5
clear:
