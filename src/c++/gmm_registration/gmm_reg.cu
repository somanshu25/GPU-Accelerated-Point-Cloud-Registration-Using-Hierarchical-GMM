#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "gmm_reg.h"
#include "../common/utilities.h"
#include "gmm_reg_kernels.h"


/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1


GMMRegistration::GMMRegistration(int K) {
	this->numComponents = K;
}

void GMMRegistration::initSimulation(int numSrc, glm::vec3* srcPc, int numTarget, glm::vec3* targetPc) {
	// allocate src pc and src trans pc
	this->numSrcPc = numSrc;
	cudaMalloc((void**)&(this->dev_srcPc), numSrc * sizeof(glm::vec3));
	cudaMalloc((void**)&(this->dev_srcTransPc), numSrc * sizeof(glm::vec3));

	// allocate target pc
	this->numTargetPc = numTarget;
	cudaMalloc((void**)&(this->dev_targetPc), numTarget * sizeof(glm::vec3));

	// allocate src GMM mu & target GMM mu
	cudaMalloc((void**)&(this->dev_srcMu), numComponents * sizeof(glm::vec3));
	cudaMalloc((void**)&(this->dev_targetMu), numComponents * sizeof(glm::vec3));
	// allocate src GMM psi & target GMM psi
	cudaMalloc((void**)&(this->dev_srcPsi), numComponents * sizeof(float));
	cudaMalloc((void**)&(this->dev_targetPsi), numComponents * sizeof(float));

	checkCUDAErrorWithLine("cudaMalloc failed!", __LINE__);

	cudaMemcpy(dev_srcPc, srcPc, numSrc * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_targetPc, targetPc, numTarget * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	checkCUDAErrorWithLine("cudaMecmpy failed!", __LINE__);

}

void GMMRegistration::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numSrcPc + numTargetPc + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO<< <fullBlocksPerGrid, blockSize >>>(numSrcPc, dev_srcPc, numTargetPc, dev_targetPc, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO<< <fullBlocksPerGrid, blockSize >>>(numSrcPc, numTargetPc, vbodptr_velocities, scene_scale);
	checkCUDAErrorWithLine("copyBoidsToVBO failed!", __LINE__);
	cudaDeviceSynchronize();
}

void GMMRegistration::pointCloudRegisterGPU(float dt) {

}


void GMMRegistration::endSimulation() {


	cudaFree(dev_srcPc);
	cudaFree(dev_targetPc);
	cudaFree(dev_srcTransPc);

	cudaFree(dev_srcMu);
	cudaFree(dev_targetMu);

	cudaFree(dev_srcPsi);
	cudaFree(dev_targetPsi);

	checkCUDAErrorWithLine("cudaFree failed!", __LINE__);
}