#include "Neuron.h"
#include <cmath>

//
// Created by Shadows on 7/6/2021.
//

Neuron::Neuron(uint outputs, uint idx) {
    for (uint i = 0; i < outputs; i++) {
        outputWeights.push_back(Connection());
        outputWeights.back().weight = random() / double(RAND_MAX) * 0.1;
    }
    this->idx = idx;
}

void Neuron::setOutput(double output) {
    this->output = output;
}

double Neuron::getOutput() const {
    return output;
}

double Neuron::activationFunc(double x) {
    return tanh(x);
}

double Neuron::activationDeriv(double x) {
    return 1.0 - x * x;
}

void Neuron::forwardProp(const Layer &prev) {
    double sum = 0;
    for (const Neuron &n : prev) {
        sum += n.getOutput() * n.outputWeights[idx].weight;
    }

    output = activationFunc(sum);
}

void Neuron::calcOutGradient(double target) {
    double delta = target - output;
    gradient = delta * activationDeriv(output);
}

double Neuron::sumDOW(const Layer &layer) {
    double sum = 0;
    for (uint i = 0; i < layer.size() - 1; i++) {
        sum += outputWeights[i].weight * layer[i].gradient;
    }
    return sum;
}

void Neuron::calcGradient(const Layer &next) {
    double dow = sumDOW(next);
    gradient = dow * activationDeriv(output);
}

void Neuron::updateWeights(Layer &prev) {
    for (uint i = 0; i < prev.size(); i++) {
        Neuron &n = prev[i];
        double oldDelta = n.outputWeights[idx].dWeight;
        double newDelta = learningRate * n.getOutput() * gradient + momentumRate * oldDelta;

        n.outputWeights[idx].dWeight = newDelta;
        n.outputWeights[idx].weight += newDelta;
    }
}