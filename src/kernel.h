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

namespace scanMatchingICP {
    void initSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target);
    void cpuNaive(vector<glm::vec3>& source, vector<glm::vec3>& target,int iter);
    void gpuImplement(bool KDtree,int iter);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();
    void unitTest();
}
