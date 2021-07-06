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
     * The size of this array should be 1 + L
     * Topology Format:
     * idx 0 : Number of layers (L) in the network.
     * idx 1 : Number of inputs
     * idx [2,N-1] : Number of neurons in a hidden layer.
     * idx N : Number of outputs
     */
    NeuralNet(const vector<uint> &topology);

    void forwardProp(const vector<double> &input);

    void backProp(const vector<double> &output);

    void getResults(vector<double> &results) const;

private:
    vector<Layer> layers;
    double error;
    double recentAvgError;
    double recentSmoothingFactor;
};

#endif //NNTEST_NEURALNET_H
