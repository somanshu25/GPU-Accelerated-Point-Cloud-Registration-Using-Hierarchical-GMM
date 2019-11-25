#include <iostream>
#include "pointdata.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

PointData::PointData(string filename) {
	cout << "Reading data points from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		throw;
	}
	while (fp_in.good()) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (tokens.size() != 3)
				continue;
			points.push_back(glm::vec3(atof(tokens[0].c_str()), atof(tokens[1].c_str()), atof(tokens[2].c_str())));
		}
	}
}
