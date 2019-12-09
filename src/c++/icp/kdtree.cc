#include <vector>
#include<stdio.h>
#include "kdtree.h"
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <algorithm>

using namespace std;

bool sortbyX(const glm::vec3 &a, const glm::vec3 &b) {
	return a.x < b.x;
}

bool sortbyY(const glm::vec3 &a, const glm::vec3 &b) {
	return a.y < b.y;
}

bool sortbyZ(const glm::vec3 &a, const glm::vec3 &b) {
	return a.z < b.z;
}

void KDtree::createTree(vector<glm::vec3>& target,glm::vec4 *result) {
	insertNode(target,result,0,0,-1,0,target.size()-1);
}

void KDtree::insertNode(vector<glm::vec3>& value, glm::vec4 *result, int pos, int depth, int parent, int start, int end) {

	if (start > end)
		return;

	if (depth % 3 == 0) {
		// Sort by X
		sort(value.begin()+start, value.begin()+end+1, sortbyX);
	}
	else if (depth % 3 == 1) {
		// Sort by Y
		sort(value.begin()+start, value.begin()+end+1, sortbyY);
	}
	else {
		// Sort by Z
		sort(value.begin()+start, value.begin()+end+1, sortbyZ);
	}

	int mid = (start + end)/2;
	result[pos] = glm::vec4(value[mid].x, value[mid].y, value[mid].z, 1.0f);

	// Insert into the left child elements
	insertNode(value, result, (2 * pos) + 1, depth+1, pos, start, mid - 1);
	// Insert into the right child elements
	insertNode(value, result, (2 * pos) + 2, depth+1, pos, mid + 1, end);

}

KDtree::Node::Node(int pos,int d, bool b) {
	index = pos;
	depth = depth;
	bad = b;
}

KDtree::Node::Node() {
	index = -1;
	depth = -1;
	bad = false;
	parentPos = -1;
}
