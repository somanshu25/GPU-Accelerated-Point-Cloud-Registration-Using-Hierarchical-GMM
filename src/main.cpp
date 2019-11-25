/**
 * @file      main.cpp
 * @brief     Scan Matching ICP
 * @authors   Somanshu Agarwal
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string.h> 
//#include "svd3.h"
//#include "svd3_cuda.h"
#include "main.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vector>
#include <sstream>
#include "glm/glm.hpp"

using namespace std;

#define VISUALIZE 1
#define cpuVersion 0
#define gpuVersion 0
#define gpuKDTree 0
#define singleGMM 1

glm::vec3 rotation(-0.5f, 0.5f, 0.3f);
glm::vec3 translate(0.1f, 0.1f, 0.2f);
glm::vec3 scale(1.5f, 1.5f, 1.5f);
glm::mat4 transformed = utilityCore::buildTransformationMatrix(translate, rotation, scale);

glm::vec3 rotationTar(-1.0f, 0.1f, 0.3f);
glm::vec3 translateTar(0.1f, 0.2f, 0.2f);
glm::vec3 scaleTar(1.5f, 1.5f, 1.5f);
glm::mat4 transformedTar = utilityCore::buildTransformationMatrix(translateTar, rotationTar, scaleTar);

int N_FOR_VIS;
int iter = 0;
vector<glm::vec3> sourcePoints;
vector<glm::vec3> targetPoints;

void readData(string filename, vector<glm::vec3>& points , vector<glm::vec3>& pointsTar) {
	glm::vec3 point;
	int count = 0;
	cout << "Reading data points from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	ifstream fp_in;
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		throw;
	}
	while (fp_in.good()) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty()) {
			count++;
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (count < 25)
				continue;
			if (tokens.size() != 3)
				break;
			point = glm::vec3(atof(tokens[0].c_str()), atof(tokens[1].c_str()), atof(tokens[2].c_str()));
			points.push_back(glm::vec3(transformed * glm::vec4(point, 1)));

			if (points.size() < 5) {
				printf("%.4f %.4f\n", point.x, points[points.size() - 1].x);
			}
				

			pointsTar.push_back(glm::vec3(transformedTar * glm::vec4(point, 1)));
			//points.push_back(transformed * glm::vec3(atof(tokens[0].c_str()), atof(tokens[1].c_str()), atof(tokens[2].c_str())));
		}
	}
}

void readData2(string filename, vector<glm::vec3>& points) {
	int count = 0;
	cout << "Reading data points from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	ifstream fp_in;
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		throw;
	}
	while (fp_in.good()) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty()) {
			count++;
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (count < 25)
				continue;
			if (tokens.size() != 3)
				break;
			points.push_back(glm::vec3(atof(tokens[0].c_str()), atof(tokens[1].c_str()), atof(tokens[2].c_str())));
		}
	}
}
int main2(int argc, char* argv[]) {
	
	readData("../data/bun000.ply", sourcePoints, targetPoints);
	//readData2(argv[1], sourcePoints);
	//readData2(argv[2], targetPoints);

	//readData2("../data-set/cone.txt", sourcePoints);
	printf("For Source Points, first 5 points are: \n");
	for (int i = 0; i < 5; i++)
		printf(" %0.4f, %0.4f, %0.4f \n", sourcePoints[i].x, sourcePoints[i].y, sourcePoints[i].z);

	N_FOR_VIS = sourcePoints.size() + targetPoints.size();
	//N_FOR_VIS = sourcePoints.size();
	for (int i = 0; i < 4; i ++)
		for (int j = 0; j < 4 ; j++)
			printf("The value of matrix[%d][%d] is: %0.4f \n",j,i,transformed[j][i]);

	printf("Size of source pointcloud: %d\n", sourcePoints.size());
	printf("Size of target pointcloud: %d\n", targetPoints.size());

	if (init(argc, argv)) {
		mainLoop();
		scanMatchingICP::endSimulation();
		return 0;
	}
	else {
		return 1;
	}
}


//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
	// Set window title to "Student Name: [SM 2.0] GPU Name"
	cudaDeviceProp deviceProp;
	int gpuDevice = 0;
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (gpuDevice > device_count) {
		std::cout
			<< "Error: GPU device number is greater than the number of devices!"
			<< " Perhaps a CUDA-capable GPU is not installed?"
			<< std::endl;
		return false;
	}
	cudaGetDeviceProperties(&deviceProp, gpuDevice);
	int major = deviceProp.major;
	int minor = deviceProp.minor;

	//std::ostringstream ss;
	//ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
	deviceName = "Somanshu Window";

	// Window setup stuff
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		std::cout
			<< "Error: Could not initialize GLFW!"
			<< " Perhaps OpenGL 3.3 isn't available?"
			<< std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	// Initialize drawing state
	initVAO();

	// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
	// change the device ID.
	cudaGLSetGLDevice(0);

	cudaGLRegisterBufferObject(boidVBO_positions);
	cudaGLRegisterBufferObject(boidVBO_velocities);

	// Initialize N-body simulation
	scanMatchingICP::initSimulation(sourcePoints, targetPoints);

	updateCamera();

	initShaders(program);

	glEnable(GL_DEPTH_TEST);

	return true;
}

void initVAO() {

	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < N_FOR_VIS; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}


	glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
	glGenBuffers(1, &boidVBO_positions);
	glGenBuffers(1, &boidVBO_velocities);
	glGenBuffers(1, &boidIBO);

	glBindVertexArray(boidVAO);

	// Bind the positions array to the boidVAO by way of the boidVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Bind the velocities array to the boidVAO by way of the boidVBO_velocities
	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
	glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(velocitiesLocation);
	glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void initShaders(GLuint * program) {
	GLint location;

	program[PROG_BOID] = glslUtility::createProgram(
		"shaders/boid.vert.glsl",
		"shaders/boid.geom.glsl",
		"shaders/boid.frag.glsl", attributeLocations, 2);
	glUseProgram(program[PROG_BOID]);

	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

//====================================
// Main loop
//====================================
void runCUDA(int iter) {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
	// use this buffer

	float4 *dptr = NULL;
	float *dptrVertPositions = NULL;
	float *dptrVertVelocities = NULL;

	cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
	cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

	
	// execute the kernel
	
	#if singleGMM
		scanRegistration::runSimulation(sourcePoints, targetPoints);
	#elif cpuVersion
		scanMatchingICP::cpuNaive(sourcePoints, targetPoints,iter);
	#elif gpuVersion
		scanMatchingICP::gpuImplement(gpuKDTree,iter);
	#endif
		

	#if VISUALIZE
		scanMatchingICP::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
	#endif

	// unmap buffer object
	cudaGLUnmapBufferObject(boidVBO_positions);
	cudaGLUnmapBufferObject(boidVBO_velocities);
}

void mainLoop() {
	double fps = 0;
	double timebase = 0;
	int frame = 0;
	//scanMatchingICP::unitTest(); // LOOK-1.2 We run some basic example code to make sure
					   // your CUDA development setup is ready to go.

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		iter++;
		frame++;
		double time = glfwGetTime();

		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}

		runCUDA(iter);

		//if (iter == 2)
		//	break;

		std::ostringstream ss;
		ss << "[";
		ss.precision(1);
		ss << std::fixed << fps;
		ss << " fps] " << deviceName;
		glfwSetWindowTitle(window, ss.str().c_str());

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
		glUseProgram(program[PROG_BOID]);
		glBindVertexArray(boidVAO);
		glPointSize((GLfloat)pointSize);
		glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);

		glfwSwapBuffers(window);
#endif
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}


void errorCallback(int error, const char *description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera();
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
		updateCamera();
	}

	lastX = xpos;
	lastY = ypos;
}

void updateCamera() {
	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.z = zoom * cos(theta);
	cameraPosition.y = zoom * cos(phi) * sin(theta);
	cameraPosition += lookAt;

	projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;

	GLint location;

	glUseProgram(program[PROG_BOID]);
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}
