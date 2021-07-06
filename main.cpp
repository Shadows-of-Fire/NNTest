#include "NeuralNet.h"
#include <iostream>

int main() {

    vector<uint> topology = {2, 4, 1};
    NeuralNet net(topology);

    for (int i = 0; i < 2000; i++) {
        vector<double> input;

        input.push_back((random() / double(RAND_MAX) > 0.5) ? 1 : 0);
        input.push_back((random() / double(RAND_MAX) > 0.5) ? 1 : 0);
        net.forwardProp(input);

        vector<double> output;
        output.push_back((input[0] == input[1]) ? 0 : 1);
        net.backProp(output);

        vector<double> results;
        net.getResults(results);
        cout << "Iteration " << i << endl;
        cout << "In : (" << input[0] << "," << input[1] << ")" << endl;
        cout << "Out : " << results[0] << endl;
        cout << "Expected : " << output[0] << endl;
    }

    getchar();

    return 0;
}
