#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <vector>
#include <cuda.h>
#include <cstdio>
#include<cuda.h>
#include "gmm_kernels.h"
#include "gmm.h"
#include "utilities.h"

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
dim3 threadsPerBlock(blockSize);

// Buffers allocated for the logic
glm::vec3 *dev_points;
glm::vec3 *dev_mu;
glm::mat3 *dev_covar;
float *dev_logPriors;
float *dev_logProb;

__device__ float calculateMahalanobisDistance(glm::vec3 a, glm::vec3 b, glm::mat3 covar) {
	glm::mat3 covarInv = glm::inverse(covar);
	glm::vec3 temp = (a - b) * covarInv;
	float distance = glm::dot(a - b, temp);
	return distance;
}

__device__ float calculateMahalanobisDistance(glm::vec2 a, glm::vec2 b, glm::mat2 covar) {
	glm::mat2 covarInv = glm::inverse(covar);
	glm::vec2 temp = (a - b) * covarInv;
	float distance = glm::dot(a - b, temp);
	return distance;
}

__device__ float calculateProbability(glm::vec3 mean, glm::mat3 covar, glm::vec3 point) {
	int dim = 3;
	float value = -0.5*(dim*log(TWO_PI) + log(glm::determinant(covar)) + calculateMahalanobisDistance(point, mean, covar));
	return value;
}

__device__ float calculateProbability(glm::vec2 mean, glm::mat2 covar, glm::vec2 point) {
	int dim = 2;
	float value = -0.5*(dim*log(TWO_PI) + log(glm::determinant(covar)) + calculateMahalanobisDistance(point, mean, covar));
	return value;
}

__global__ void expectationStep(glm::vec3 *data,glm::vec3 *mean, glm::mat3 *covar, float *priors, float *prob, int N, int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	float sum = 0;
	for (int i = 0; i < components; i++) {
		prob[index*components + i] = log(priors[i]) + log(calculateProbability(mean[i], covar[i], data[index]));
		sum = sum + exp(prob[index*components + i]);
	}

	sum = log(sum);
	for (int i = 0; i < components; i++) {
		prob[index*components + i] /= sum;
	}
}

/*
__global__ void maximizationStep(glm::vec3 *data, glm::vec3 *mean, glm::mat3 *covar, float *priors, float *prob, int N) {
	
}
*/
__global__ void expectationStep(glm::vec2 *data, glm::vec2 *mean, glm::mat2 *covar, float *priors, float *prob, int N,int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	float sum = 0;
	for (int i = 0; i < components; i++) {
		prob[index*components + i] = log(priors[i]) + log(calculateProbability(mean[i], covar[i], data[index]));
		sum = sum + exp(prob[index*components + i]);
	}

	sum = log(sum);
	for (int i = 0; i < components; i++) {
		prob[index*components + i] /= sum;
	}
}

/*
_global__ void maximizationStep(glm::vec2 *data, glm::vec2 *mean, glm::mat2 *covar, float *priors, float *logProb, int N) {

	

}
*/
void GMM::solve(vector<glm::vec3> points, glm::vec3 *mu, glm::mat3 *covar, int iterations, int N) {

	float *logPriors = new float[components];

	for (int i = 0; i < components; i++) {
		logPriors[i] = log(1.0 / components);
	}

	float *prob = new float[N * components];

	int numObjects = N;
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	cudaMalloc((void**)&dev_points, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_points failed!");

	cudaMalloc((void**)&dev_mu, components * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_mu failed!");

	cudaMalloc((void**)&dev_covar, components * sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMalloc dev_covar failed!");

	cudaMalloc((void**)&dev_logPriors, components * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_logPriors failed!");

	cudaMalloc((void**)&dev_logProb, numObjects * components * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_logPriors failed!");

	cudaMemcpy(dev_points, &points[0], N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_points failed!");

	cudaMemcpy(dev_mu, &mu[0], N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_points failed!");

	cudaMemcpy(dev_covar, &covar[0], N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_points failed!");

	for (int i = 0; i < iterations; i++) {
		expectationStep <<<fullBlocksPerGrid, blockSize >>> (dev_points, dev_mu, dev_covar, dev_logPriors, dev_logProb, N, components);
		checkCUDAErrorWithLine("Kernel expectation Step failed!");

		//maximizationStep<<<fullBlocksPerGrid, blockSize >>>(points, mu, covar, logPriors, logProb, N, components);
		//checkCUDAErrorWithLine("Kernel maximization Step failed!");
	}
}

/*
void GMM::solve(glm::vec2 *points, glm::vec2 *mu, glm::mat2 *covar, int iterations, int N) {

	float *priors = new float[components];

	for (int i = 0; i < components; i++) {
		priors[i] = 1.0 / components;
	}

	float *prob = new float[N * components];

	int numObjects = N;
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	cudaMalloc((void**)&points, numObjects * sizeof(glm::vec2));

	for (int i = 0; i < iterations; i++) {
		expectationStep<<<>>>(points, mu, covar, priors, prob, N,components);
		//maximizationStep(points, mu, covar, priors, prob, N);
	}
}
*/
void scanRegistration::runSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target) {
	numObjects = source.size() + target.size();
	int sourceSize = source.size();
	int targetSize = target.size();

	GMM g1(100);
	
	glm::vec3 *mu = new glm::vec3[100];
	glm::mat3 *covar = new glm::mat3[100];

	g1.solve(source, mu, covar, 50, sourceSize);

	//glm::vec3 *mu2 = new glm::vec3[100];
	//glm::mat3 *covar2 = new glm::mat3[100];

	//g1.solve(target, mu2, covar2, 50, targetSize);
}