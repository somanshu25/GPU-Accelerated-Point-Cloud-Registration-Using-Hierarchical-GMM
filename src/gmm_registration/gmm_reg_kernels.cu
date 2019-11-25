#include "gmm_reg_kernels.h"

__global__ void kernCopyPositionsToVBO(int N_SRC, glm::vec3 *pos_src, int N_TARGET, glm::vec3 *pos_target, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N_SRC + N_TARGET) {
		if (index < N_SRC) {
			vbo[4 * index + 0] = pos_src[index].x * c_scale;
			vbo[4 * index + 1] = pos_src[index].y * c_scale;
			vbo[4 * index + 2] = pos_src[index].z * c_scale;
			vbo[4 * index + 3] = 1.0f;
		}
		else {
			vbo[4 * index + 0] = pos_target[index - N_SRC].x * c_scale;
			vbo[4 * index + 1] = pos_target[index - N_SRC].y * c_scale;
			vbo[4 * index + 2] = pos_target[index - N_SRC].z * c_scale;
			vbo[4 * index + 3] = 1.0f;
		}
	}
}

__global__ void kernCopyVelocitiesToVBO(int N_SRC, int N_TARGET, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N_SRC + N_TARGET) {
		if (index < N_SRC) {
			vbo[4 * index + 0] = 1.0 + 0.3f;
			vbo[4 * index + 1] = 1.0 + 0.3f;
			vbo[4 * index + 2] = 1.0 + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
		else {
			vbo[4 * index + 0] = 1.0 + 0.3f;
			vbo[4 * index + 1] = 0.0 + 0.3f;
			vbo[4 * index + 2] = 0.0 + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
	}
}

