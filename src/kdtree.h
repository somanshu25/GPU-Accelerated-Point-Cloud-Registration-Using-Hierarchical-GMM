#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <vector>
#include <cuda.h>
#include <cstdio>
using namespace std;

namespace KDtree {
	
	class Node {
		public:
			Node();
			Node(int pos,int depth,bool bad);
			bool bad;
			int depth;
			int index;
			int parentPos;
	};
	
	void createTree(vector<glm::vec3>& target,glm::vec4 *result);
	void insertNode(vector<glm::vec3>& value,glm::vec4 *result, int pos, int depth,int parent,int start, int end);
	//int calculateMaxDepth(vector<glm::vec3> value, int depth, int maxDepth);
}
