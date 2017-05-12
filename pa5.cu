#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include "device_launch_parameters.h"

extern "C"{
#include "ppmFile.h"
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Keep x and y values within the boundaries of the image
__device__ __host__
int checkBoundsHeight(int val,int height) {

  if (val < 0){
    val = 0;
  } else if (val > height-1){
    val = height-1;
  }

  return val;
}

// Keep x and y values within the boundaries of the image
__device__ __host__
int checkBoundsWidth(int val, int width){

  if (val < 0){
    val = 0;
  } else if (val > width-1){
    val = width-1;
  }

  return val;
}

__device__ __host__
void ImageSetPixelDevice(Image *image, int x, int y, int chan, unsigned char val){
  int offset = (y * image->width + x) * 3 + chan;

  image->data[offset] = val;
}

__device__ __host__
unsigned  char ImageGetPixelDevice(Image *image, int x, int y, int chan){
  int offset = (y * image->width + x) * 3 + chan;

  return image->data[offset];
}

// Calculates new image color for blurring
__device__ __host__
int calcBlurColour(Image *imageFile, int minX, int maxX, int minY, int maxY, int channel){
  //pixelsAdded = (maxX - minX)*(maxY-minY);
  int pixelsAdded = 0;
  long val = 0;

  for (int x = minX; x <= maxX; x++){
    for (int y = minY; y <= maxY; y++){
      val = val + ImageGetPixelDevice(imageFile,x,y,channel);
      pixelsAdded++;
    }
  }

  if (pixelsAdded <= 0){
    pixelsAdded = 1;
  }

  val = val/pixelsAdded;

  return val;
}

__global__
void blurImage(unsigned char *data, unsigned char *blurData, int height, int width, int r,int totalThreads){

  int myID = (blockIdx.z * gridDim.x * gridDim.y +
              blockIdx.y * gridDim.x +
              blockIdx.x) * blockDim.x +
              threadIdx.x;
  int minX;
  int maxX;
  int minY;
  int maxY;

  unsigned char newR = 0;

  unsigned char newG = 0;
  unsigned char newB = 0;

  Image *image = (Image *) malloc(sizeof(Image));
  image->data   = data;
  image->width  = width;
  image->height = height;

  for (int y = 0; y < height; y++){
    for (int x = myID; x < width; x = x + totalThreads){
      image->data = data;
      // Calculates new bounds for the blurring of the pixel based on the blur radius
      minX = checkBoundsWidth(x - r,image->width);
      maxX = checkBoundsWidth(x + r,image->width);
      minY = checkBoundsHeight(y - r,image->height);
      maxY = checkBoundsHeight(y + r,image->height);

      newR = calcBlurColour(image, minX, maxX, minY, maxY,0);
      newG = calcBlurColour(image, minX, maxX, minY, maxY,1);
      newB = calcBlurColour(image, minX, maxX, minY, maxY,2);

      image->data = blurData;

      ImageSetPixelDevice(image, x, y, 0, newR);
      ImageSetPixelDevice(image, x, y, 1, newG);
      ImageSetPixelDevice(image, x, y, 2, newB);
    }
  }
}

__global__
void pixelateImage(unsigned char *data, unsigned char *blurData, int height, int width, int r,int totalThreads){

  int myID = (blockIdx.z * gridDim.x * gridDim.y +
              blockIdx.y * gridDim.x +
              blockIdx.x) * blockDim.x +
              threadIdx.x;
  int minX;
  int maxX;
  int minY;
  int maxY;

  int spacing = r*2;

  unsigned char newR = 0;

  unsigned char newG = 0;
  unsigned char newB = 0;

  Image *image = (Image *) malloc(sizeof(Image));
  image->data   = data;
  image->width  = width;
  image->height = height;

  for (int y = r; y < height; y = y + spacing){
    for (int x = r; x < width; x = x + spacing){
      image->data = data;
      // Calculates new bounds for the blurring of the pixel based on the blur radius
      minX = checkBoundsWidth(x - r,image->width);
      maxX = checkBoundsWidth(x + r,image->width);
      minY = checkBoundsHeight(y - r,image->height);
      maxY = checkBoundsHeight(y + r,image->height);

      newR = calcBlurColour(image, minX, maxX, minY, maxY,0);
      newG = calcBlurColour(image, minX, maxX, minY, maxY,1);
      newB = calcBlurColour(image, minX, maxX, minY, maxY,2);

      image->data = blurData;

      for (int x = minX; x < maxX; x++){
        for (int y = minY; y < maxY; y++){
          // Blurs 3 channels (r,g,b) of pixel
          ImageSetPixelDevice(image,x,y,0,newR);
          ImageSetPixelDevice(image,x,y,1,newG);
          ImageSetPixelDevice(image,x,y,2,newB);
        }
      }
    }
  }
}

int main(int argc, char**argv){
  Image *imageFile, *outputImage;
  unsigned char *data,*blurData;
  int r;

  int blockHeight = 1;
  int blockWidth = 1024;

  int gridHeight = 1;
  int gridWidth = 1;
  int gridLength = 1;

  int totalThreads = blockHeight*blockWidth*gridHeight*gridWidth*gridLength;

  dim3 block(blockHeight,blockWidth);
  dim3 grid(gridHeight,gridWidth,gridLength);

  // For clocking time
  time_t start,end1,end2;


  imageFile = ImageRead(argv[2]);
  outputImage = ImageRead(argv[2]);
  r = strtol(argv[1],NULL,10);
  const size_t imageSize = sizeof(int)*2 + imageFile->height*imageFile->width*3;

  data = (unsigned char*) malloc(imageFile->height*imageFile->width*3);
  blurData = (unsigned char*) malloc(imageFile->height*imageFile->width*3);

  // Allocate unified memory for cpu and gpu
  gpuErrchk(cudaMallocManaged(&data,imageFile->height*imageFile->width*3));
  gpuErrchk(cudaMallocManaged(&blurData,imageFile->height*imageFile->width*3));
  memcpy(data,imageFile->data,imageFile->height*imageFile->width*3);
  //memcpy(blurData,imageFile->data,imageFile->height*imageFile->width*3);

  printf("Blurring Image based on GPU...\n");

  // Start Timer
  start = clock();
  printf("%s\n",argv[4]);
  if (strcmp(argv[4],"blur") == 0){
    blurImage<<<gridHeight,blockHeight*blockWidth>>>(data,blurData,imageFile->height,imageFile->width,r,totalThreads);
    cudaDeviceSynchronize();
  } else {
    pixelateImage<<<gridHeight,blockHeight*blockWidth>>>(data,blurData,imageFile->height,imageFile->width,r,totalThreads);
    cudaDeviceSynchronize();
  }

  // References blurred data to output image
  outputImage->data = blurData;

  // End Timer
  end1 = clock() - start;

  ImageWrite(outputImage,argv[3]);
  printf("Image: %s blurred, saved as: %s\n",argv[2],argv[3]);
  printf("Clock ticks: %li\n",end1);

  //endTime = MPI_Wtime();
  //totalTime = endTime - startTime;
  //printf("Took %f seconds\n", totalTime);

  cudaFree(data);
  cudaFree(blurData);
  return 0;
}
