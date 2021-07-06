#include "NeuralNet.h"
#include <iostream>
#include <cmath>

//
// Created by Shadows on 7/6/2021.
//

/**
 * Creates a new NeuralNet
 * The size of this array should be 1 + L
 * Topology Format:
 * idx 0 : Number of layers (L) in the network.
 * idx 1 : Number of inputs
 * idx [2,N-1] : Number of neurons in a hidden layer.
 * idx N : Number of outputs
 */
NeuralNet::NeuralNet(const vector<uint> &topology) {
    uint nLayers = topology.size();
    for (uint i = 0; i < nLayers; i++) {
        layers.push_back(Layer());
        uint nNeurons = topology[i];
        //<= due to the additional bias neuron per-layer.  Number of outputs is size of next layer, or 0, if output.
        for (int j = 0; j <= nNeurons; j++) {
            layers.back().push_back(Neuron((i == nLayers - 1) ? 0 : topology[i + 1], j));
            cout << "Made neuron " << j << endl;
        }

        layers.back().back().setOutput(1);
    }
}

void NeuralNet::forwardProp(const vector<double> &input) {
    for (uint i = 0; i < input.size(); i++) {
        layers[0][i].setOutput(input[i]);
    }
    for (uint l = 1; l < layers.size(); l++) {
        Layer &prev = layers[l - 1];
        for (uint n = 0; n < layers[l].size(); n++) {
            layers[l][n].forwardProp(prev);
        }
    }
}

void NeuralNet::backProp(const vector<double> &output) {
    Layer &outLayer = layers.back();
    error = 0.0;
    for (uint i = 0; i < outLayer.size() - 1; i++) {
        double delta = output.at(i) - outLayer[i].getOutput();
        error += delta * delta;
    }
    error /= outLayer.size() - 1;
    error = sqrt(error);

    recentAvgError = (recentAvgError * recentSmoothingFactor + error) / (recentSmoothingFactor + 1);

    for (uint i = 0; i < outLayer.size() - 1; i++) {
        outLayer[i].calcOutGradient(output.at(i));
    }

    for (uint i = layers.size() - 2; i > 0; i--) {
        Layer &hidden = layers[i];
        Layer &next = layers[i + 1];
        for (uint n = 0; n < hidden.size(); n++) {
            hidden[n].calcGradient(next);
        }
    }

    for (uint i = layers.size() - 1; i > 0; i--) {
        Layer &layer = layers[i];
        Layer &prev = layers[i - 1];
        for (uint n = 0; n < layer.size() - 1; n++) {
            layer[n].updateWeights(prev);
        }
    }
}

void NeuralNet::getResults(vector<double> &results) const {
    results.clear();
    for (uint i = 0; i < layers.back().size() - 1; i++) {
        results.push_back(layers.back()[i].getOutput());
    }
}
