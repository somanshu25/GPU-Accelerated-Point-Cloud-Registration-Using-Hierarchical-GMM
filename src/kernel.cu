#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilities.h"
#include "kernel.h"
#include "svd3.h"
#include <thrust/reduce.h>
#include "kdtree.h"
#include <glm/vec3.hpp>
#include <chrono>
#include <ctime>
#include <ratio>

#define RECORD_TIMING 1
//#include <glm/gtx/string_cast.hpp>
using namespace std;

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
int sourceSize;
int targetSize;
int kdTreeLength;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *devCorrespond;
glm::vec3 *devTempSource;
glm::mat3 *devMult;
glm::vec4 *devKDtree;
KDtree::Node *devStackNode;
// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__global__ void kernResetVec3Buffer(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}


__global__ void calculateCorrespondPoint(int sourceSize,int targetSize, glm::vec3 *devPos, glm::vec3 *correspond) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= sourceSize)
		return;

	float min_dist = glm::distance(devPos[index],devPos[sourceSize]);
	int id = 0;
	float dist;
	for (int i = 1; i < targetSize; i++) {
		dist = glm::distance(devPos[index], devPos[i+sourceSize]);
		if (dist < min_dist) {
			min_dist = dist;
			id = i;
		}
	}
	correspond[index] = devPos[id+sourceSize];
}

__global__ void meanCentrePoints(int N, glm::vec3 *tmpSource, glm::vec3 *correspond, glm::vec3 meanSource, glm::vec3 meanCorrespond) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	tmpSource[index] -= meanSource;
	correspond[index] -= meanCorrespond;

}

__device__ int getBranch(int depth,glm::vec3 point, glm::vec3 ref) {
	//if (depth == 0)
		//printf("The point inside consideration is: %0.4f, %0.4f, %0.4f \n", ref.x, ref.y, ref.z);
	if (depth % 3 == 0) {
		//printf("The x-point inside consideration is: %0.4f and %0.4f and the decision inside is: %d\n", point.x, ref.x, (point.x < ref.x));
		return (point.x < ref.x);
	}


	if (depth % 3 == 1) {
		//printf("The y-point inside consideration is: %0.4f and %0.4f and the decision inside is: %d\n", point.y, ref.y, (point.y < ref.y));
		return (point.y < ref.y);
	}


	else {
		//printf("The z-point inside consideration is: %0.4f and %0.4f and the decision inside is: %d\n", point.z, ref.z, (point.z < ref.z));
		return (point.z < ref.z);
	}
		
}

__device__ int searchBadSide(int depth, glm::vec3 point, glm::vec3 ref, float bestdist) {
	if (depth % 3 == 0)
		return (abs(point.x - ref.x) < bestdist);

	if (depth % 3 == 1)
		return (abs(point.y - ref.y) < bestdist);

	else
		return (abs(point.z - ref.z) < bestdist);
}

/*
__device__ int traverseTree(glm::vec3 point,int depth,int nodePos, glm::vec4 *result,int bestPos) {
	
	//if (depth == 0)
	//	printf("Hello\n");

	if (result[nodePos].w!= 1.0f)
		return bestPos;

	float dist = glm::distance(glm::vec3(result[nodePos]), point);
	float bestDistance = glm::distance(glm::vec3(result[bestPos]) , point);

	if (dist < bestDistance)
		bestPos = nodePos;
	//printf("The point under consideration is: %0.4f, %0.4f, %0.4f and the distance between the point :%0.4f, %0.4f, %0.4f is %0.4f for depth of %d and the branch prediction is %d\n", point.x, point.y, point.z, result[nodePos].x, result[nodePos].y, result[nodePos].z, dist, depth, getBranch(depth, point, glm::vec3(result[nodePos])));
	//Node *good, Node *bad;
	int goodNodePos, badNodePos;
	if (getBranch(depth, point, glm::vec3(result[nodePos]))) {
		goodNodePos = 2* nodePos + 1;
		badNodePos = 2 * nodePos + 2;
	}
	else {
		goodNodePos = 2* nodePos + 2;
		badNodePos = 2 * nodePos + 1;
	}
	bestPos = traverseTree(point, depth+1, goodNodePos, result,bestPos);

	if (searchBadSide(depth, glm::vec3(result[nodePos]), point, bestDistance))
		bestPos = traverseTree(point, depth+1, badNodePos, result,bestPos);

	return bestPos;

}
*/

__global__ void findCorrespondenceKD(int sourceSize,glm::vec3 *dev_pos, glm::vec4 *result,glm::vec3 *correspond,int totalNum,int maxDepth, KDtree::Node n0, KDtree::Node goodNode, KDtree::Node badNode, KDtree::Node *stackNode) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= sourceSize)
		return;

	int top = 0;
	int depth = 0;
	int bestPos = 0;
	int nodePos;
	//KDtree::Node n0 = KDtree::Node(0,0,true);
	
	n0.index = 0;
	n0.bad = false;
	n0.depth = 0;
	n0.parentPos = -1;
	
	//KDtree::Node *stackNode = new KDtree::Node[totalNum];
	stackNode[index*maxDepth + top] = n0;
	int goodNodePos, badNodePos;
	float dist, bestDistance;
	glm::vec3 point = dev_pos[index];
	while (top != -1) {

		// do A POP ON THE STACK
		KDtree::Node n = stackNode[index*maxDepth + top--];

		// Check if it is NULL (or actually going to child which does not exist
		if (result[n.index].w == 0.0f)
			continue;

		// Check if it is a bad node
		if (n.bad) {
			bestDistance = glm::distance(glm::vec3(result[bestPos]),point);
			if (!(searchBadSide(n.depth-1, glm::vec3(result[n.parentPos]), point, bestDistance)))
				continue;
		}

		nodePos = n.index;
		depth = n.depth;

		dist = glm::distance(glm::vec3(result[nodePos]), point);
		bestDistance = glm::distance(glm::vec3(result[bestPos]), point);

		if (dist < bestDistance)
			bestPos = nodePos;

		if (getBranch(depth, point, glm::vec3(result[nodePos]))) {
			goodNodePos = 2 * nodePos + 1;
			badNodePos = 2 * nodePos + 2;
		}
		else {
			goodNodePos = 2 * nodePos + 2;
			badNodePos = 2 * nodePos + 1;
		}
		//KDtree::Node goodNode = KDtree::Node(goodNodePos, depth + 1, false);
		//KDtree::Node badNode = KDtree::Node(badNodePos, depth + 1, true);
		//KDtree::Node goodNode, badNode;
		
		goodNode.index = goodNodePos;
		goodNode.depth = depth + 1;
		goodNode.bad = false;
		goodNode.parentPos = nodePos;

		badNode.index = badNodePos;
		badNode.depth = depth + 1;
		badNode.bad = true;
		badNode.parentPos = nodePos;
		
		stackNode[index*maxDepth + ++top] = badNode;
		stackNode[index*maxDepth + ++top] = goodNode;
	}

	correspond[index] = glm::vec3(result[bestPos]);
}
/**
* Initialize memory, update some globals
*/
void scanMatchingICP::initSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target) {
  
  numObjects = source.size() + target.size();
  sourceSize = source.size();
  targetSize = target.size();
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kdTreeLength = pow(2.0, ceil(log2(targetSize / 1.0) / 1.0) + 1.0);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, numObjects * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&devCorrespond, sourceSize * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc devCorrespond failed!");

  cudaMalloc((void**)&devTempSource, sourceSize * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc devTempSource failed!");

  cudaMalloc((void**)&devMult, sourceSize * sizeof(glm::mat3));
  checkCUDAErrorWithLine("cudaMalloc devTempSource failed!");

  cudaMalloc((void**)&devKDtree, kdTreeLength * sizeof(glm::vec4));
  checkCUDAErrorWithLine("cudaMalloc devTempSource failed!");

  cudaMalloc((void**)&devStackNode, sourceSize * (ceil(log2(targetSize / 1.0) / 1.0) + 1.0) * sizeof(KDtree::Node));
  checkCUDAErrorWithLine("cudaMalloc devTempSource failed!");


  //int depth = KDtree::calculateMaxDepth(target,0,0);

  glm::vec4 *result = new glm::vec4[kdTreeLength];

  KDtree::createTree(target,result);
  cudaMemcpy(devKDtree, result, kdTreeLength * sizeof(glm::vec4), cudaMemcpyHostToDevice);
  
  printf("Data points are: \n");
  for (int i = 0; i < 20; i++) {
	  printf("%0.4f, %0.4f, %0.4f \n", result[i].x, result[i].y, result[i].z);
  }

  printf("The total number of elements are: %d and in binary tree array is: %d \n", targetSize, kdTreeLength);
  
  // Initialize the KDtree
 // cudaMalloc((void**)&devKD, sourceSize * sizeof(glm::));
  //checkCUDAErrorWithLine("cudaMalloc devTempSource failed!");

  // copy both scene and target to output points
   cudaMemcpy(dev_pos, &source[0], source.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
   cudaMemcpy(&dev_pos[source.size()], &target[0], target.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

   kernResetVec3Buffer << <dim3((source.size() + blockSize - 1) / blockSize), blockSize >> > (source.size(), dev_vel1, glm::vec3(1, 1, 1));
   kernResetVec3Buffer << <dim3((target.size() + blockSize - 1) / blockSize), blockSize >> > (target.size(), &dev_vel1[source.size()], glm::vec3(1, 1, 0));

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}


__global__ void outerProduct(int sourceSize, glm::vec3 *source, glm::vec3 *target,glm::mat3 *out) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= sourceSize)
		return;

	out[index] = glm::outerProduct(source[index], target[index]);
}

__global__ void kernMatrixMultiplication(glm::vec3 *M, glm::vec3 *N, float *Out, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("The values of m , n and k are :%d , %d %d \n", m, n , k);
	//printf("The values of row and col are: %d & %d \n", row, col);
	float sum = 0;
	float a, b;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			a = (row == 0 ? N[i].x : row == 1 ? N[i].y : N[i].z);
			b = (col == 0 ? M[i].x : col == 1 ? M[i].y : M[i].z);
			sum += a * b;
			//printf("hello the value of Sum is : %0.3f\n",sum);
		}
		//printf("The values are %d & %d \n", row, col);
		Out[row*k + col] = sum;
		//printf("The value is: %0.2f \n", Out[row*k + col]);
	}

}

__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < cols && idy < rows)
	{
		unsigned int pos = idy * cols + idx;
		unsigned int trans_pos = idx * rows + idy;
		mat_out[trans_pos] = mat_in[pos];
	}
}

__global__ void updatePoints(int sourceSize,int targetSize,glm::vec3 *dev_pos,glm::mat3 R, glm::vec3 trans) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= sourceSize)
		return;
	
	dev_pos[index] = R * dev_pos[index] + trans;
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void scanMatchingICP::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

void scanMatchingICP::cpuNaive(vector<glm::vec3>& source, vector<glm::vec3>& target,int iter) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
	
	
	vector<glm::vec3> sourceCorrespond, sourceNew;
	
	/*
	printf("Hello here I came in iteration:%d\n",iter);
	
	for (int i = 0; i < 5; i++) {
		printf("%0.4f %0.4f, %0.4f \n", source[i].x, source[i].y, source[i].z);
	}

	printf("For Target Points, first 5 points are: \n");

	for (int i = 0; i < 5; i++) {
		printf("%0.4f %0.4f, %0.4f \n", target[i].x, target[i].y, target[i].z);
	}
	*/
	#if RECORD_TIMING
		using namespace std::chrono;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
	#endif

	int index;
	float dist = 0;
	float min_dist = FLT_MAX;
	for (int i = 0; i < source.size(); i++) {
		min_dist = FLT_MAX;
		for (int j = 0; j < target.size(); j++) {
			dist = glm::distance(source[i], target[j]);
			if (dist < min_dist) {
				min_dist = dist;
				index = j;
			}
		}
		sourceCorrespond.push_back(target[index]);
	}

	/*
	printf("For Correspondance Points in target, first 5 points are: \n");

	for (int i = 0; i < 10; i++) {
		printf("%0.4f %0.4f, %0.4f \n", sourceCorrespond[i].x, sourceCorrespond[i].y, sourceCorrespond[i].z);
	}
	*/
	// Mean of the traget and new Ones

	glm::vec3 meanSource(0.0f, 0.0f, 0.0f);
	glm::vec3 meanCorrespond(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < source.size(); i++) {
		meanSource += source[i];
		meanCorrespond += sourceCorrespond[i];
	}

	meanSource /= source.size();
	meanCorrespond /= source.size();

	//printf("Mean of source Points: %0.4f, %0.4f, %0.4f\n", meanSource.x, meanSource.y, meanSource.z);
	//printf("Mean of correspondence Points: %0.4f, %0.4f, %0.4f\n", meanCorrespond.x, meanCorrespond.y, meanCorrespond.z);

	glm::vec3 point;

	for (int i = 0; i < source.size(); i++) {
		point = source[i] - meanSource;
		sourceNew.push_back(point);
		sourceCorrespond[i] = sourceCorrespond[i] - meanCorrespond;
	}

	float W[3][3] = { 0 };
	float a, b;

	float sum = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			sum = 0;
			for (int k = 0; k < source.size(); k++) {
				a = (i == 0 ? sourceCorrespond[k].x : i == 1 ? sourceCorrespond[k].y : sourceCorrespond[k].z);
				b = (j == 0 ? sourceNew[k].x : j == 1 ? sourceNew[k].y : sourceNew[k].z);
				sum += a * b;
			}
			W[i][j] = sum;
		}
	}

	/*
	printf("The Values of Matrx Multiplication are: \n");
	
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ",W[i][j]);
		}
		printf("\n");
	}
	*/
	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
	);

	glm::mat3 g_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 g_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	// Get transformation from SVD
	glm::mat3 R = g_U * g_Vt;
	glm::vec3 t = meanCorrespond - R * meanSource;

	/*
	printf("The values of U are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", g_U[i][j]);
		}
		printf("\n");
	}
	
	printf("The values of Vt are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", g_Vt[i][j]);
		}
		printf("\n");
	}

	printf("The Values of Rotation Matrix are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", R[i][j]);
		}
		printf("\n");
	}

	printf("The translational Matrix is: %0.4f, %0.4f, %0.4f\n", t.x, t.y, t.z);
	*/
	// update source points
	for (int i = 0; i < source.size(); i++) {
		source[i] = R * source[i] + t;
	}	
	#if RECORD_TIMING
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		if (iter < 10)
			std::cout << "For iter " << iter << " " << time_span.count() << " seconds." << endl;
	#endif
	//cudaMemcpy(dev_pos, &source[0], source.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pos, &source[0], source.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	//kernResetVec3Buffer << <dim3((source.size() + blockSize - 1) / blockSize), blockSize >> > (source.size(), dev_vel1, glm::vec3(1, 1, 1));
	//kernResetVec3Buffer << <dim3((target.size() + blockSize - 1) / blockSize), blockSize >> > (target.size(), &dev_vel1[source.size()], glm::vec3(1, 1, 0));

}

void scanMatchingICP::gpuImplement(bool Kdtree,int iter) {
	
	//float *W = new float[9];
	dim3 fullBlocksPerGrid((sourceSize + blockSize - 1) / blockSize);

	#if RECORD_TIMING
		using namespace std::chrono;
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
	#endif

	KDtree::Node n0, goodNode, badNode;
	if (Kdtree) {
		findCorrespondenceKD << <fullBlocksPerGrid, blockSize >> > (sourceSize, dev_pos, devKDtree, devCorrespond, numObjects, (ceil(log2(targetSize / 1.0) / 1.0) + 1.0), n0, goodNode, badNode, devStackNode);
		checkCUDAErrorWithLine("Kernel CorrespondPoint KD failed!");
	}
	else {
		calculateCorrespondPoint << <fullBlocksPerGrid, blockSize >> > (sourceSize, targetSize, dev_pos, devCorrespond);
		checkCUDAErrorWithLine("Kernel CorrespondPoint failed!");
	}
	
	glm::vec3 meanSource(0.0f, 0.0f, 0.0f);
	glm::vec3 meanCorrespond(0.0f, 0.0f, 0.0f);

	thrust::device_ptr<glm::vec3> sourcePtr(dev_pos);
	thrust::device_ptr<glm::vec3> correspondPtr(devCorrespond);

	meanSource = glm::vec3(thrust::reduce(sourcePtr, sourcePtr + sourceSize, glm::vec3(0.0f, 0.0f, 0.0f)));
	meanCorrespond = glm::vec3(thrust::reduce(correspondPtr, correspondPtr + sourceSize, glm::vec3(0.0f, 0.0f, 0.0f)));

	meanSource /= sourceSize;
	meanCorrespond /= sourceSize;

	//printf("Mean of source Points: %0.4f, %0.4f, %0.4f\n", meanSource.x, meanSource.y, meanSource.z);
	//printf("Mean of correspondence Points: %0.4f, %0.4f, %0.4f\n", meanCorrespond.x, meanCorrespond.y, meanCorrespond.z);

	/*
	glm::vec3 *check3 = new glm::vec3[sourceSize];
	cudaMemcpy(check3, devCorrespond, sourceSize * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	printf("Correspondence Data points are: \n");
	for (int i = 0; i < sourceSize; i++) {
		printf("%0.4f, %0.4f, %0.4f \n", check3[i].x, check3[i].y, check3[i].z);
	}
	*/

	cudaMemcpy(devTempSource,dev_pos,sourceSize * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	meanCentrePoints << <fullBlocksPerGrid, blockSize >> > (sourceSize,devTempSource, devCorrespond, meanSource, meanCorrespond);
	checkCUDAErrorWithLine("Kernel meanCentrePoints failed!");
	
	outerProduct << <fullBlocksPerGrid, blockSize >> > (sourceSize, devTempSource, devCorrespond,devMult);
	checkCUDAErrorWithLine("Kernel outerProduct failed!");

	//printf("There are more shit\n");
	/*
	glm::vec3 *check = new glm::vec3[sourceSize];
	cudaMemcpy(check, devCorrespond, sourceSize * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	glm::vec3 *check2 = new glm::vec3[sourceSize];
	cudaMemcpy(check2, devTempSource, sourceSize * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	printf("Mean centered Data points are: \n");
	for (int i = 0; i < sourceSize; i++) {
		printf("%0.4f, %0.4f, %0.4f \n", check2[i].x, check2[i].y, check2[i].z);
	}

	printf("Mean centered Correspondence Data points are: \n");
	for (int i = 0; i < sourceSize; i++) {
		printf("%0.4f, %0.4f, %0.4f \n", check[i].x, check[i].y, check[i].z);
	}
	*/

	glm::mat3 W = thrust::reduce(thrust::device,devMult, devMult + sourceSize, glm::mat3(0));
	//printf("There are more and more shit\n");
	/*
	printf("The Values of Matrx Multiplication are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", W[i][j]);
		}
		printf("\n");
	}
	*/

	//kernMatrixMultiplication << <fullBlocksPerGrid, blockSize >> > (devTempSource, devCorrespond,W,3,sourceSize,3);
	//checkCUDAErrorWithLine("Kernel Matrix Multiplication failed!");
	
	
	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][0],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
	);

	glm::mat3 g_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 g_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	// Get transformation from SVD
	glm::mat3 R = g_U * g_Vt;
	glm::vec3 t = meanCorrespond - R * meanSource;

	updatePoints << <fullBlocksPerGrid, blockSize >> > (sourceSize,targetSize,dev_pos,R,t);
	checkCUDAErrorWithLine("Kernel updatePoints failed!");
	
	cudaDeviceSynchronize();

	#if RECORD_TIMING
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		if (iter < 30)
			std::cout << time_span.count() << endl;
	#endif
	/*printf("The Values of Rotation Matrix are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", R[i][j]);
		}
		printf("\n");
	}

	printf("The translational Matrix is: %0.4f, %0.4f, %0.4f\n", t.x, t.y, t.z);
	*/
	//printf("The rotation Matrix is")
	//free(check);
}

void scanMatchingICP::endSimulation() {
  cudaFree(dev_vel1);
  //cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(devCorrespond);
  cudaFree(devTempSource);
  cudaFree(devKDtree);
  cudaFree(devStackNode);
  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void scanMatchingICP::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
