//
// Created by Shadows on 7/6/2021.
//
#include "Utils.h"

#ifndef NNTEST_NEURON_H
#define NNTEST_NEURON_H


class Neuron {
public:
    /**
     * Creates a new Neuron (a single NN unit)
     * @param outputs The number of outputs this neuron has.  Must be == # neurons in the next layer (0 for output)
     * @param idx The index of this neuron in the layer it exists within.
     */
    Neuron(uint outputs, uint idx);

    /**
     * Sets the output of this neuron.
     * @param output
     */
    void setOutput(double output);

    double getOutput() const;

    double activationFunc(double x);

    double activationDeriv(double x);

    void forwardProp(const Layer &prev);

    void calcOutGradient(double target);

    double sumDOW(const Layer &layer);

    void calcGradient(const Layer &next);

    void updateWeights(Layer &prev);

private:
    double output;
    vector<Connection> outputWeights;
    uint idx;
    double gradient;
    constexpr static double learningRate = 0.15, momentumRate = 0.05;
};

#endif //NNTEST_NEURON_H
