#include <helper_image.h>
#include <cuda_runtime.h>

#include "stdio.h"

#define N 5

__global__ void median_filter(float* dLena, int width, int height, float* dResult){

    int fSize = 3;
    int currCount, currX, currY, i, j, k, l, flag, offset, index_x, x_count, index_y, y_count, X_BOUND = width-1, Y_BOUND = height-1, fTotal = fSize*fSize;
    float *filter = new float[fTotal];
    
    float temp;

    //For each of the threads in the kernel
    for(i=0; i < blockDim.x; ++i){
        index_x = blockIdx.x * i + threadIdx.x;
	for(j=0; j < blockDim.y;++j){	
            index_y = blockIdx.y * j + threadIdx.y;
            currCount = 0;
	    //Calculate the offset for the current median filter
	    offset = (fSize/2)*-1;    

	    //for each value in the filter
            for (k = index_x+offset, x_count=0; x_count < fSize; ++k, ++x_count) {
                //Logic to handle the edge cases
		currX = k;
                if(currX < 0){
                    currX = currX * -1;
                } else if (currX > X_BOUND) {
                    currX = currX - X_BOUND;
                    currX = X_BOUND - currX;
                }
                for (l = index_y+offset, y_count=0; y_count < fSize; ++l, ++y_count ) {
                    currY = l;
                    if(currY < 0){
                        currY = currY * -1;
                    } else if (currY > Y_BOUND) {
                        currY = currY - Y_BOUND;
                        currY = Y_BOUND - currY;
                    }
			
		    //Add this value to the median filter array
                    filter[currCount] = *(dLena+((width*currX)+currY));
                    ++currCount;

                }
            }
 
            //Bubble sort
	    flag=1;
            for(k=1; (k<=fTotal) && flag;++k){
                flag=0;
                for(l=0;l<(fTotal-1);++l){
                    if(filter[l+1] > filter[l]){
                        temp = filter[l];
                        filter[l] = filter[l+1];
                        filter[l+1] = temp;
                        flag = 1;
                    }
                }
            }
	
	//Add value to the result
	*(dResult+((width*index_y)+index_x)) = filter[(fTotal/2)];
	//Print to see value in results after its set
	//printf("%d,%d is now %f\n",index_x, index_y,*(dResult+((width*index_y)+index_x)));
        }
    }

}
int main(int argc, char** argv){

	cudaEvent_t l_start;
	cudaEvent_t l_stop;

	cudaEventCreate( &l_start );
	cudaEventCreate( &l_stop );

	float* lena = NULL, *dLena=NULL;
	float *result=NULL, *dResult=NULL, *goldStd=NULL;
	unsigned int width, height;
	char* path = sdkFindFilePath("lena.pgm", argv[0]);

	cudaEventRecord( l_start, 0 );
	//Load Lena
	if(!sdkLoadPGM(path, &lena, &width, &height)){
		printf("Unsuccessful\n");
		return -1;
	}
	cudaEventRecord( l_stop, 0 );
	
	float timeValue;
	cudaEventElapsedTime( &timeValue, l_start, l_stop );

	std::cout<<"Load time: "<<timeValue<<std::endl;

	// Calculate block and grid sizes
        dim3 block_size;
        block_size.x = 4;
        block_size.y = 4;

        dim3 grid_size;
        grid_size.x = width / block_size.x;
        grid_size.y = height / block_size.y;

	//Malloc memory
	cudaMalloc((void**)&dLena, (width*height)*sizeof(float));
	cudaMalloc((void**)&dResult, (width*height)*sizeof(float));
	result = (float*)malloc(sizeof(float)*(width*height));
	goldStd = (float*)malloc(sizeof(float)*(width*height));

	//Copy image results for device
	cudaMemcpy(dLena, lena, (width*height)*sizeof(float), cudaMemcpyHostToDevice);

	//printf("Result:\n");

	//printf("{CudaPrintfInt => %s}\n",cudaGetErrorString(cudaPrintfInit()));

	cudaEvent_t d_start;
        cudaEvent_t d_stop;

        cudaEventCreate( &d_start );
        cudaEventCreate( &d_stop );

	//call the kernels to calculate median filter 
	cudaEventRecord( d_start, 0 );
	median_filter<<<grid_size,block_size>>>(dLena, width, height, dResult);
	cudaEventRecord( d_stop, 0 );

	float d_time;
	cudaEventElapsedTime( &d_time, d_start, d_stop );

        std::cout<<"Calculate time: "<<d_time<<std::endl;

	//printf("{cudaPrintfDisplay => %s}\n",cudaGetErrorString(cudaPrintfDisplay()));

	int i,j;

	cudaMemcpy(result, dResult, (width*height)*sizeof(float), cudaMemcpyDeviceToHost);

	//Print out to see return value from kernels
//	for(i=0;i<height;++i){
//                for(j=0;j<width;++j){
//                        printf("TOM:%f\n", *result);
//                }
//        }

	int X_BOUND = 511;
	int Y_BOUND = 511;

	int fSize = 3;

    int currCount, currX, currY, k, l, flag, offset, index_x, x_count, index_y, y_count, fTotal = fSize*fSize;
    float *filter = (float*)malloc(sizeof(float)*fTotal);
    float temp;

    //Settings for gold standard
    //Same logic for Kernel
    for(i=0; i < width; ++i){
        for(j=0; j < height;++j){
            index_x = i;
            index_y = j;
            currCount = 0;
            offset = (fSize/2)*-1;

            for (k = index_x+offset, x_count=0; x_count < fSize; ++k, ++x_count) {
                currX = k;
                if(currX < 0){
                    currX = currX * -1;
                } else if (currX > X_BOUND) {
                    currX = currX - X_BOUND;
                    currX = X_BOUND - currX;
                }
                for (l = index_y+offset, y_count=0; y_count < fSize; ++l, ++y_count ) {
                    currY = l;
                    if(currY < 0){
                        currY = currY * -1;
                    } else if (currY > Y_BOUND) {
                        currY = currY - Y_BOUND;
                        currY = Y_BOUND - currY;
                    }
                    filter[currCount] = *(lena + ((currX*width)+currY));
                    ++currCount;
		}
            }

            flag=1;
            for(k=1; (k<=fTotal) && flag;++k){
                flag=0;
                for(l=0;l<(fTotal-1);++l){
                    if(filter[l+1] > filter[l]){
                        temp = filter[l];
                        filter[l] = filter[l+1];
                        filter[l+1] = temp;
                        flag = 1;
                    }
                }
            }
		
            //SET LOCATION to filter[(fTotal/2)];
		*(goldStd+((width*index_y)+index_x))=filter[(fTotal/2)];
        }
    }

	float total=0;
	for(i=0; i<height;++i){
		for(j=0; j<width; ++j){
			//total += *(goldStd+((width*j)+i))-*(result+((width*j)+i));	
			//printf("%d,%d-%f\n", i, j, *(result+((width*i)+j)));
		}
	}
	//printf("Difference:%f\n\n", total);

	return 0;
}
