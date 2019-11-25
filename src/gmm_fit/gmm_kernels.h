#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>
using namespace std;

namespace scanRegistration {
	void initSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target,int components);
	void runSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target);
	void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
	void endSimulation();
}
