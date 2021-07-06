#include <vector>

//
// Created by Shadows on 7/6/2021.
//

using namespace std;

#ifndef NNTEST_UTILS_H
#define NNTEST_UTILS_H

typedef unsigned int uint;

struct Connection {
    double weight;
    double dWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

#endif //NNTEST_UTILS_H
