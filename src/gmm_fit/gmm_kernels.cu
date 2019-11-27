#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "../common/utilities.h"
#include <thrust/reduce.h>
#include <glm/vec3.hpp>
#include <chrono>
#include <ctime>
#include <ratio>
#include "gmm_kernels.h"
#include "gmm.h"
#include <thrust/execution_policy.h>

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1f

glm::vec3 *dev_points;
glm::vec3 *dev_posvel;

int numPoints;
//int components;
int sourcePoints;
int targetPoints;
//#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/

void printArrayFloat(int n, float *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%0.4f ", a[i]);
	}
	printf("]\n");
}

void printArrayFloatExp(int n, float *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%0.4f ", exp(a[i]));
	}
	printf("]\n");
}

__global__ void kernResetVec3Buffer2(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO2(int N, glm::vec3 *pos, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO2(int N, glm::vec3 *vel, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}


__device__ float calculateMahalanobisDistance(glm::vec3 a, glm::vec3 b, glm::mat3 covar) {
	glm::mat3 covarInv = glm::inverse(covar);
		
	glm::vec3 temp = (a - b) * covarInv;
	float distance = glm::dot(a - b, temp);
	//if (index == 400) {
	//	printf("\n Distance is: %0.5f", distance);
	//}
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

__global__ void expectationStep(glm::vec3 *data, glm::vec3 *mean, glm::mat3 *covar, float *logPriors, float *prob, int N, int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	float sum = 0;
	for (int i = 0; i < components; i++) {
		prob[index*components + i] = logPriors[i] + calculateProbability(mean[i], covar[i], data[index]);
		sum = sum + exp(prob[index*components + i]);
	}
	sum = log(sum);

	for (int i = 0; i < components; i++) {
		prob[index*components + i] = prob[index*components + i] - sum;
	}
}

__global__ void calculateTotalComponent(float *out, float *in, int N, int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= components)
		return;
	float sum = 0;
	for (int i = 0; i < N; i++) {
		sum += exp(in[i*components + index]);
	}
	out[index] = log(sum);
}

__global__ void calculateTotalComponentSum(float *out, float *in1, float *in2, int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= components)
		return;

	out[index] = exp(in1[index] + in2[index]);
}

__global__ void updateLogPriors(float *a, float *b, float accum,int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= components)
		return;

	b[index] = b[index] + a[index] - log(accum);
}

__global__ void updateMu(glm::vec3 *data, float *totalComponentPrior, float *logProb, glm::vec3 *out,int components, int N) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= components)
		return;

	glm::vec3 sum(0.0f,0.0f,0.0f);
	for (int i = 0; i < N; i++) {
		sum += data[i] * exp(logProb[i*components + index]);
	}
	out[index] = sum / (exp(totalComponentPrior[index]));
}

__global__ void updateCovar(glm::vec3 *data,float *totalComponentPrior, float *logProb, glm::vec3 *mean, glm::mat3 *covar,int components, int N) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= components)
		return;

	glm::mat3 sum(0.0f);
	for (int i = 0; i < N; i++) {
		sum += glm::outerProduct(data[i] - mean[index], data[i] - mean[index]) * exp(logProb[i*components + index]);
	}
	covar[index] = sum / (exp(totalComponentPrior[index]));

}

__global__ void logLikelihoodValueComponents(glm::vec3 *data, glm::vec3 *mean, glm::mat3 *covar ,float *dev_logPriors, float *dev_PriorsSum, int components, int N) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= components)
		return;

	float sum = 0;
	for (int i = 0; i < components; i++) {
		sum += exp(dev_logPriors[i]) * exp(calculateProbability(mean[i], covar[i], data[index]));
	}
	dev_PriorsSum[index] = log(sum);
}

__global__ void expectationStep(glm::vec2 *data, glm::vec2 *mean, glm::mat2 *covar, float *priors, float *prob, int N, int components) {
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
__global__ void distancefromCentroids(glm::vec3 *points, glm::vec3 *mu, float *distance, int N, int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	int index_min;
	float dist_min = 999;
	for (int i = 0; i < components; i++) {
		distance[index*components + i] = glm::distance(points[index], mu[i])*glm::distance(points[index], mu[i]);
		if (distance[index*components + i] < dist_min) {
			index_min = i;
			dist_min = distance[index*components + i];
		}
	}


}

__global__ void assignPoint(glm::vec3 *points,float* distance, glm::vec3 *dev_pointsAssigned,int N, int components) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	for (int i = 0; i < components; i++) {

	}
}
*/
/*
_global__ void maximizationStep(glm::vec2 *data, glm::vec2 *mean, glm::mat2 *covar, float *priors, float *logProb, int N) {



}
*/

void maximizationStep(glm::vec3 *data, glm::vec3 *mean, glm::mat3 *covar, float *logPriors, float *logProb, int N,int components) {

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGrid1((components + blockSize - 1) / blockSize);

	float *dev_totalComponentPrior;
	float *dev_totalComponentPriorSumExp;

	float *weights3 = new float[components];

	cudaMalloc((void**)&dev_totalComponentPrior, components * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_totalComponentPrior failed!");

	cudaMalloc((void**)&dev_totalComponentPriorSumExp, components * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_totalComponentSum failed!");
	
	calculateTotalComponent << <fullBlocksPerGrid1, blockSize >> > (dev_totalComponentPrior,logProb,N,components);
	checkCUDAErrorWithLine("kernel calculateTotalComponent failed");
	
	cudaDeviceSynchronize();

	calculateTotalComponentSum << <fullBlocksPerGrid1, blockSize >> > (dev_totalComponentPriorSumExp, dev_totalComponentPrior, logPriors, components);
	checkCUDAErrorWithLine("kernel calculateTotalComponentSum failed");

	float accum = thrust::reduce(thrust::device, dev_totalComponentPriorSumExp, dev_totalComponentPriorSumExp + components);
	
	updateLogPriors << <fullBlocksPerGrid1,blockSize >> > (dev_totalComponentPrior,logPriors,accum,components);
	checkCUDAErrorWithLine("kernel updateLogPriors failed");

	float *weights = new float[components];
	
	updateMu << <fullBlocksPerGrid1, blockSize >> > (data,dev_totalComponentPrior, logProb, mean,components,N);
	checkCUDAErrorWithLine("kernelupdateMus failed");

	cudaDeviceSynchronize();
	updateCovar << <fullBlocksPerGrid1, blockSize >> > (data, dev_totalComponentPrior, logProb, mean,covar, components, N);
	checkCUDAErrorWithLine("kernel updateCovar failed");
	
	cudaFree(dev_totalComponentPrior);
}
/*
void kmeans(glm::vec3 *points,glm::vec3 *mu, int N, int components) {
	
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGrid1((components + blockSize - 1) / blockSize);
	float *dev_distance;

	cudaMalloc((void**)&dev_distance, components*N * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_distance failed!");

	distancefromCentroids << <fullBlocksPerGrid, blockSize >> (points,mu,dev_distance,N,components);
	checkCUDAErrorWithLine("Kernel distancefromCentroids failed!");

	assignPoint << <fullBlocksPerGrid, blockSize >> > (dev_points,distance,dev_pointsAssigned,N,components);
	checkCUDAErrorWithLine("Kernel assignPoint failed!");

	calculateMeans << <fullBlocksPerGrid, blockSize >> > (dev_points, distance, dev_pointsAssigned, N, components);
	checkCUDAErrorWithLine("Kernel assignPoint failed!");
}
*/
void GMM::solve(vector<glm::vec3> points, glm::vec3 *mu, float *weights, int iterations, int N) {

	printf("\n The randomized mean values are: \n");
	for (int i = 0; i < components; i++) {
		int p = rand();
		mu[i] = points[p % sourcePoints];
		weights[i] = 1.0 / components;
		printf(" The index founs is %d, and it's point is :%0.4f, %0.4f, %0.4f\n", p, mu[i].x, mu[i].y, mu[i].z);
	}

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGrid1((components + blockSize - 1) / blockSize);

	float *logPriors = new float[components];
	glm::vec3 *dev_mu;
	glm::mat3 *dev_covar;
	float *dev_logPriors;
	float *dev_logProb;
	float *dev_PriorsSum;
	//glm::vec3 *dev_source;

	for (int i = 0; i < components; i++) {
		logPriors[i] = log(weights[i]);
	}

	glm::mat3 m3(1.0f);
	glm::mat3 *covar = new glm::mat3[components];

	for (int i = 0; i < components; i++) {
		covar[i] = m3;
	}

	//printf("\nHello\n");

	//cudaMalloc((void**)&dev_source, N * sizeof(glm::vec3));
	//checkCUDAErrorWithLine("cudaMalloc dev_source failed!");

	cudaMalloc((void**)&dev_mu, components * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_mu failed!");

	//printf("\nIt came here");

	cudaMalloc((void**)&dev_covar, components * sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMalloc dev_covar failed!");

	cudaMalloc((void**)&dev_logPriors, components * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_logPriors failed!");

	cudaMalloc((void**)&dev_logProb, N * components * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_logProb failed!");

	cudaMalloc((void**)&dev_PriorsSum, N * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_covar failed!");

	//cudaMemcpy(dev_source, &points[0], N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	//checkCUDAErrorWithLine("cudaMemCpy dev_source failed!");

	cudaMemcpy(dev_mu, &mu[0], components * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_mu failed!");

	cudaMemcpy(dev_covar, &covar[0], components * sizeof(glm::mat3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_covar failed!");

	cudaMemcpy(dev_logPriors, &logPriors[0], components * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_logpriors failed!");

	//printArrayFloat(components, logPriors);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	/*
	int kmeans_iterations = 20;

	for (int i = 0; i < kmeans_iterations; i++) {
		kmeans(dev_points,mu,N,components);
	}
	*/
	for (int i = 0; i < iterations; i++) {
		expectationStep << <fullBlocksPerGrid, blockSize >> > (dev_points, dev_mu, dev_covar, dev_logPriors, dev_logProb, N, components);
		checkCUDAErrorWithLine("Kernel expectation Step failed!");
		//printf("\nprinted here");
		cudaDeviceSynchronize();


		maximizationStep(dev_points, dev_mu, dev_covar, dev_logPriors, dev_logProb, N, components);

		logLikelihoodValueComponents << <fullBlocksPerGrid, blockSize >> > (dev_points,dev_mu,dev_covar,dev_logPriors,dev_PriorsSum,components,N);
		checkCUDAErrorWithLine("Kernel logLikelihoodValueComponents failed!");

		float logLikelihoodvalue = thrust::reduce(thrust::device, dev_PriorsSum, dev_PriorsSum + N);

		float *weights2 = new float[components];

		cudaMemcpy(weights2, dev_logPriors, components * sizeof(float), cudaMemcpyDeviceToHost);

		for (int j = 0; j < components; j++) {
			weights2[j] = exp(weights2[j]);
		}

		printf("\nAfter iteration %d, the weights are :\n",i+1);
		//printArrayFloat(components, weights2);
		float sumWeights = thrust::reduce(thrust::host, weights2, weights2 + components);
		printf("\n The sum of wights equal to %0.5f:\n",sumWeights);
		printf("\nThe Log Likeliihood value is: %0.5f \n", logLikelihoodvalue);
		cudaDeviceSynchronize();
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Time elapsed: %.4f\n", milliseconds);

	cudaMemcpy(mu, dev_mu, components * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemCpy dev_mu to host failed!");
	cudaMemcpy(weights, dev_logPriors, components * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemCpy devlogPriors to Host failed!");


	for (int i = 0; i < components; i++) {
		weights[i] = exp(weights[i]);
	}

	cudaFree(dev_covar);
	cudaFree(dev_mu);
	cudaFree(dev_logProb);
	cudaFree(dev_logPriors);
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

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void scanRegistration::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numPoints + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO2 << <fullBlocksPerGrid, blockSize >> > (numPoints, dev_points, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO2 << <fullBlocksPerGrid, blockSize >> > (numPoints, dev_posvel, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}


void scanRegistration::initSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target, int components) {
	
	sourcePoints = source.size();
	targetPoints = target.size();
	numPoints = sourcePoints + targetPoints;

	dim3 fullBlocksPerGrid((numPoints + blockSize - 1) / blockSize);

	cudaMalloc((void**)&dev_points, numPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_points failed!");

	cudaMalloc((void**)&dev_posvel, numPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_posvel failed!");

	//cudaMalloc((void**)&dev_covar, components * sizeof(glm::mat3));
	//checkCUDAErrorWithLine("cudaMalloc dev_covar failed!");

	cudaMemcpy(dev_points, &source[0], sourcePoints * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_points for source failed!");

	cudaMemcpy(&dev_points[sourcePoints], &target[0], targetPoints * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_points for target failed!");

	/*
	cudaMemcpy(dev_mu_target, &mu_target[0], components * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_mu failed!");

	cudaMemcpy(dev_phi_target, &phi_target[0], components * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemCpy dev_covar failed!");
	*/
	kernResetVec3Buffer2 << <dim3((source.size() + blockSize - 1) / blockSize), blockSize >> > (source.size(), dev_posvel, glm::vec3(1, 1, 1));
	kernResetVec3Buffer2 << <dim3((target.size() + blockSize - 1) / blockSize), blockSize >> > (target.size(), &dev_posvel[source.size()], glm::vec3(1, 1, 0));

	cudaDeviceSynchronize();
}

void scanRegistration::runSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target) {

	int components = 50;
	GMM g1(components);

	glm::vec3 *mu = new glm::vec3[components];
	float *weights = new float[components];

	g1.solve(source, mu, weights, 30, sourcePoints);

	//glm::vec3 *mu2 = new glm::vec3[100];
	//glm::mat3 *covar2 = new glm::mat3[100];

	//g1.solve(target, mu2, covar2, 50, targetSize);
	//cudaMemcpy(dev_points, &source[0], source.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

void scanRegistration::endSimulation() {
	cudaFree(dev_points);
	cudaFree(dev_posvel);
}