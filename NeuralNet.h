#include "Utils.h"
#include "Neuron.h"

//
// Created by Shadows on 7/6/2021.
//

#ifndef NNTEST_NEURALNET_H
#define NNTEST_NEURALNET_H

class NeuralNet {
public:
    /**
     * Creates a new NeuralNet
     * Topology Format:
     * The size of the vector is the number of layers.
     * idx 0 : Number of inputs
     * idx [1,N-2] : Number of neurons in a hidden layer.
     * idx N-1 : Number of outputs
     */
    NeuralNet(const vector<uint> &topology);

    /**
     * Carries out a single iteration of forward propagation.
     * @param input The inputs to the neural network.  Must have size == # inputs
     */
    void forwardProp(const vector<double> &input);

    /**
     * Carries out a single iteration of backwards propagation.
     * @param target The expected outputs from the network.  Must have size == # outputs
     */
    void backProp(const vector<double> &target);

    /**
     * Fills a vector with the results from the neural network.
     * The passed vector will be cleared before use.
     * @param results A vector to be filled with the current output of this network.
     */
    void getResults(vector<double> &results) const;

private:
    vector<Layer> layers;
    double error;
};

#endif //NNTEST_NEURALNET_H
