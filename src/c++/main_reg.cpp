#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <sstream>

#include <glm/glm.hpp>

#include <chrono>
#include <thread>

#include "gmm_registration/gmm_reg.h"
#include "main_reg.h"

#define VISUALIZE 1

const float DT = 0.2f;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.01, 0.01);

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)


glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
	glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
	return translationMat * rotationMat * scaleMat;
}


glm::vec3 *src_pc, *target_pc;
int numSrc, numTarget;

using namespace std::this_thread; // sleep_for, sleep_until
using namespace std::chrono;
using namespace GMMRegDriver;

//glm::vec3 rotation(-0.5f, 0.5f, 0.3f);
//glm::vec3 translate(0.1f, 0.1f, 0.2f);
//glm::vec3 scale(2.0f, 2.0f, 2.0f);
//glm::mat4 transformSrc = buildTransformationMatrix(translate, rotation, scale);
//
//glm::vec3 rotationTar(-1.5f, 0.5f, 0.3f);
//glm::vec3 translateTar(0.1f, 0.2f, 0.2f);
//glm::vec3 scaleTar(2.0f, 2.0f, 2.0f);
//glm::mat4 transformTar = buildTransformationMatrix(translateTar, rotationTar, scaleTar);

GMMRegistration GMMReg(800);

bool wait = false;

void printArray2D(float *X, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			printf("%.2f ", X[i*nC + j]);
		printf("\n");
	}
}

std::vector<std::string> tokenizeString(std::string str) {
	std::stringstream strstr(str);
	std::istream_iterator<std::string> it(strstr);
	std::istream_iterator<std::string> end;
	std::vector<std::string> results(it, end);
	return results;
}

std::istream& safeGetline(std::istream& is, std::string& t) {
	t.clear();

	// The characters in the stream are read one-by-one using a std::streambuf.
	// That is faster than reading them one-by-one using the std::istream.
	// Code that uses streambuf this way must be guarded by a sentry object.
	// The sentry object performs various tasks,
	// such as thread synchronization and updating the stream state.

	std::istream::sentry se(is, true);
	std::streambuf* sb = is.rdbuf();

	for (;;) {
		int c = sb->sbumpc();
		switch (c) {
		case '\n':
			return is;
		case '\r':
			if (sb->sgetc() == '\n')
				sb->sbumpc();
			return is;
		case EOF:
			// Also handle the case when the last line has no line ending
			if (t.empty())
				is.setstate(std::ios::eofbit);
			return is;
		default:
			t += (char)c;
		}
	}
}

void readPointCloud(const std::string& filename, glm::vec3 **points, int* n, bool src) {

	std::ifstream myfile(filename);
	std::string line;
	std::cout << filename << '\n';
	std::cout << myfile.is_open() << '\n';
	if (myfile.is_open()) {
		for (int i = 1; i <= 17; i++) {
			std::getline(myfile, line);
			std::cout << line << '\n';
		}

		// read element vertex
		std::getline(myfile, line);
		std::istringstream tokenStream(line);
		std::string token;
		for (int i = 1; i <= 3; i++) {
			std::getline(tokenStream, token, ' ');
		}

		int numVertex = std::stoi(token);
		*n = numVertex;
		std::cout << numVertex << '\n';

		*points = (glm::vec3 *) malloc(numVertex * sizeof(glm::vec3));

		for (int i = 1; i <= 6; i++) {
			std::getline(myfile, line);
			std::cout << line << '\n';
		}

		for (int i = 0; i < numVertex; i++) {
			std::getline(myfile, line);

			std::vector<std::string> tokens = tokenizeString(line);
			float x = atof(tokens[0].c_str());
			float y = atof(tokens[1].c_str());
			float z = atof(tokens[2].c_str());

			if (src == 0) {
				glm::vec3 p = glm::vec3(x, y, z);
				//glm::vec4 tp = transformSrc * glm::vec4(p, 1);

				(*points)[i] = glm::vec3(p);
			}
			else {
				glm::vec3 p = glm::vec3(x, y, z);
				//glm::vec4 tp = transformTar * glm::vec4(p, 1.0);
				(*points)[i] = glm::vec3(p);
			}

		}

		myfile.close();
	}
}


std::string deviceName2;
GLFWwindow *window2;

int main2(int argc, char* argv[]) {
	std::string src_filename = "../data/bun000.ply";
	std::string target_filename = "../data/bun045.ply";
	//std::string src_filename = "../data/dragon_stand/dragonStandRight_0.ply";
	//std::string target_filename = "../data/dragon_stand/dragonStandRight_48.ply";
	//std::string src_filename = "../data/happy_stand/happyStandRight_0.ply";
	//std::string target_filename = "../data/happy_stand/happyStandRight_48.ply";


	//transformSrc = buildTransformationMatrix(translate1, rotate1, scale1);
	//transformTar = buildTransformationMatrix(translate2, rotate2, scale2);

	readPointCloud(src_filename, &src_pc, &numSrc, true);
	readPointCloud(target_filename, &target_pc, &numTarget, false);
	std::cout << "Number of points: " << numSrc << ' ' << numTarget << '\n';

	glm::vec3 mean(0, 0, 0);
	for (int i = 0; i < numSrc; i++) {
		//std::cout << src[i].x << ' ' << src[i].y << ' ' << src[i].z << '\n';
		mean += src_pc[i];
	}
	mean /= numSrc;
	std::cout << mean.x << ' ' << mean.y << ' ' << mean.z << '\n';

	if (init(argc, argv)) {
		mainLoop();
		GMMReg.endSimulation();
		return 0;
	}
	else {
		return 1;
	}
}


/**
* Initialization of CUDA and GLFW.
*/
bool GMMRegDriver::init(int argc, char **argv) {
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

	std::ostringstream ss;
	//ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
	//deviceName = ss.str();
	deviceName2 = "batman";

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

	window2 = glfwCreateWindow(GMMRegDriver::width, GMMRegDriver::height, deviceName2.c_str(), NULL, NULL);
	if (!window2) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window2);
	glfwSetKeyCallback(window2, keyCallback);
	glfwSetCursorPosCallback(window2, mousePositionCallback);
	glfwSetMouseButtonCallback(window2, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	// Initialize drawing state
	initVAO();

	// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
	// change the device ID.
	cudaGLSetGLDevice(0);

	cudaGLRegisterBufferObject(GMMRegDriver::boidVBO_positions);
	cudaGLRegisterBufferObject(GMMRegDriver::boidVBO_velocities);

	// Initialize N-body simulation
	GMMReg.initSimulation(numSrc, src_pc, numTarget, target_pc);

	updateCamera();

	initShaders(GMMRegDriver::program);

	glEnable(GL_DEPTH_TEST);



	return true;
}

void GMMRegDriver::initVAO() {

	int numTotal = numSrc + numTarget;

	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (numTotal)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[numTotal] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < numTotal; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}


	glGenVertexArrays(1, &GMMRegDriver::boidVAO); // Attach everything needed to draw a particle to this
	glGenBuffers(1, &GMMRegDriver::boidVBO_positions);
	glGenBuffers(1, &GMMRegDriver::boidVBO_velocities);
	glGenBuffers(1, &GMMRegDriver::boidIBO);

	glBindVertexArray(GMMRegDriver::boidVAO);

	// Bind the positions array to the boidVAO by way of the boidVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, GMMRegDriver::boidVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * numTotal * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

	glEnableVertexAttribArray(GMMRegDriver::positionLocation);
	glVertexAttribPointer((GLuint)GMMRegDriver::positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Bind the velocities array to the boidVAO by way of the boidVBO_velocities
	glBindBuffer(GL_ARRAY_BUFFER, GMMRegDriver::boidVBO_velocities);
	glBufferData(GL_ARRAY_BUFFER, 4 * numTotal * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(GMMRegDriver::velocitiesLocation);
	glVertexAttribPointer((GLuint)GMMRegDriver::velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, GMMRegDriver::boidIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numTotal * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void GMMRegDriver::initShaders(GLuint * program) {
	GLint location;

	program[GMMRegDriver::PROG_BOID] = glslUtility::createProgram(
		"shaders/boid.vert.glsl",
		"shaders/boid.geom.glsl",
		"shaders/boid.frag.glsl", GMMRegDriver::attributeLocations, 2);
	glUseProgram(program[GMMRegDriver::PROG_BOID]);

	if ((location = glGetUniformLocation(program[GMMRegDriver::PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &GMMRegDriver::projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[GMMRegDriver::PROG_BOID], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &GMMRegDriver::cameraPosition[0]);
	}
}

//====================================
// Main loop
//====================================
void GMMRegDriver::runCUDA() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
	// use this buffer

	float4 *dptr = NULL;
	float *dptrVertPositions = NULL;
	float *dptrVertVelocities = NULL;

	cudaGLMapBufferObject((void**)&dptrVertPositions, GMMRegDriver::boidVBO_positions);
	cudaGLMapBufferObject((void**)&dptrVertVelocities, GMMRegDriver::boidVBO_velocities);

	// execute the kernel
	GMMReg.pointCloudRegisterGPU(DT);

#if VISUALIZE
	GMMReg.copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
#endif
	// unmap buffer object
	cudaGLUnmapBufferObject(GMMRegDriver::boidVBO_positions);
	cudaGLUnmapBufferObject(GMMRegDriver::boidVBO_velocities);


}

void GMMRegDriver::mainLoop() {
	double fps = 0;
	double timebase = 0;
	int frame = 0;

	while (!glfwWindowShouldClose(window2)) {
		glfwPollEvents();

		frame++;
		double time = glfwGetTime();

		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}

		runCUDA();

		std::ostringstream ss;
		ss << "[";
		ss.precision(1);
		ss << std::fixed << fps;
		ss << " fps] " << deviceName2;
		glfwSetWindowTitle(window2, ss.str().c_str());

		if (wait) {
			//sleep_for(milliseconds(5000));
			wait = false;
		}
		else {
			//sleep_for(milliseconds(700));
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
		glUseProgram(GMMRegDriver::program[GMMRegDriver::PROG_BOID]);
		glBindVertexArray(GMMRegDriver::boidVAO);
		glPointSize((GLfloat)GMMRegDriver::pointSize);
		glDrawElements(GL_POINTS, numSrc + numTarget, GL_UNSIGNED_INT, 0);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);

		glfwSwapBuffers(window2);
#endif
	}
	glfwDestroyWindow(window2);
	glfwTerminate();
}


void GMMRegDriver::errorCallback(int error, const char *description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void GMMRegDriver::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void GMMRegDriver::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	GMMRegDriver::leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	GMMRegDriver::rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void GMMRegDriver::mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (GMMRegDriver::leftMousePressed) {
		// compute new camera parameters
		GMMRegDriver::phi += (xpos - GMMRegDriver::lastX) / GMMRegDriver::width;
		GMMRegDriver::theta -= (ypos - GMMRegDriver::lastY) / GMMRegDriver::height;
		GMMRegDriver::theta = std::fmax(0.01f, std::fmin(GMMRegDriver::theta, 3.14f));
		updateCamera();
	}
	else if (GMMRegDriver::rightMousePressed) {
		GMMRegDriver::zoom += (ypos - GMMRegDriver::lastY) / GMMRegDriver::height;
		GMMRegDriver::zoom = std::fmax(0.1f, std::fmin(GMMRegDriver::zoom, 5.0f));
		updateCamera();
	}

	GMMRegDriver::lastX = xpos;
	GMMRegDriver::lastY = ypos;
}

void GMMRegDriver::updateCamera() {
	GMMRegDriver::cameraPosition.x = GMMRegDriver::zoom * sin(GMMRegDriver::phi) * sin(GMMRegDriver::theta);
	GMMRegDriver::cameraPosition.z = GMMRegDriver::zoom * cos(GMMRegDriver::theta);
	GMMRegDriver::cameraPosition.y = GMMRegDriver::zoom * cos(GMMRegDriver::phi) * sin(GMMRegDriver::theta);
	GMMRegDriver::cameraPosition += GMMRegDriver::lookAt;

	GMMRegDriver::projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;

	GLint location;

	glUseProgram(program[PROG_BOID]);
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}