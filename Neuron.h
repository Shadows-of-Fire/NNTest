//
// Created by Shadows on 7/6/2021.
//
#include "Utils.h"

#ifndef NNTEST_NEURON_H
#define NNTEST_NEURON_H


class Neuron {
public:
    Neuron(uint outputs, uint idx);

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
    vector <Connection> outputWeights;
    uint idx;
    double gradient;
    constexpr static double learningRate = 0.15, momentumRate = 0.05;
};

#endif //NNTEST_NEURON_H
