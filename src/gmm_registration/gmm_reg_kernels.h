#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

__global__ void kernCopyPositionsToVBO(int N_SRC, glm::vec3 *pos_src, int N_TARGET, glm::vec3 *pos_target, float *vbo, float s_scale);

__global__ void kernCopyVelocitiesToVBO(int N_SRC, int N_TARGET, float *vbo, float s_scale);