#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"

using namespace std;

class PointData {
private:
	ifstream fp_in;

public:
	PointData(string filename);
	~PointData();

	vector<glm::vec3> points;
};