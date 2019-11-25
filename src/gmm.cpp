#pragma once
/*
#include "gmm.h"
#include "gmm_kernels.h"

void GMM::solve(glm::vec3 *points, glm::vec3 *mu, glm::mat3 *covar,int iterations, int N) {
	
	float *priors = new float[components];

	for (int i = 0; i < components; i++) {
		priors[i] = 1.0 / components;
	}

	float *prob = new float[N * components];

	for (int i = 0; i < iterations; i++) {
		expectationStep(points, mu, covar,priors,prob,N);
		maximizationStep(points,mu,covar,priors,prob,N);
	}
}

void GMM::solve(glm::vec2 *points, glm::vec2 *mu, glm::mat2 *covar, int iterations, int N) {

	float *priors = new float[components];

	for (int i = 0; i < components; i++) {
		priors[i] = 1.0 / components;
	}

	float *prob = new float[N * components];

	for (int i = 0; i < iterations; i++) {
		expectationStep(points, mu, covar, priors, prob,N);
		maximizationStep(points, mu, covar, priors, prob,N);
	}
}
*/