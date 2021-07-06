cmake_minimum_required(VERSION 3.19)
project(NNTest)

set(CMAKE_CXX_STANDARD 20)

add_executable(NNTest main.cpp Utils.h Neuron.h Neuron.cpp NeuralNet.h NeuralNet.cpp)