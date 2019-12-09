#pragma once

#include <glm/glm.hpp>

class GMMRegistration {

public:
	int numComponents;
	int numSrcPc;
	int numTargetPc;

	glm::vec3 *dev_srcPc;
	glm::vec3 *dev_srcTransPc;
	glm::vec3 *dev_targetPc;

	glm::vec3 *dev_srcMu;
	glm::vec3 *dev_targetMu;

	float *dev_srcPsi;
	float *dev_targetPsi;


	GMMRegistration(int K);

	void initSimulation(int N1, glm::vec3* src_pc, int N2, glm::vec3* target_pc);
	
	void pointCloudRegisterGPU(float dt);
	
	void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

	void endSimulation();
};